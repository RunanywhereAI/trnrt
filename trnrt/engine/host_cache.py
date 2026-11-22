"""The host cache: a host-memory tier under the radix cache (SGLang HiCache's L2; vLLM's CPU KV offload).

When the radix cache evicts a page, its KV (every layer, this rank's heads) is copied to host
memory instead of being lost; a later request whose prefix continues past the device-resident
match into pages the host still holds gets them copied back into fresh device pages and
inserted into the tree, so it skips their prefill. Pages are identified by a chain hash over
page-aligned token blocks (vLLM prefix caching's block hash: a page's hash covers its own
tokens and its parent's hash), so a host entry matches only the exact prefix that produced it.

Write-through: a restored page keeps its host copy, so evicting it again costs nothing. The
host cache is LRU over its own capacity; pages being restored are pinned against it. Under
tensor parallelism rank 0 decides and every rank moves its own shard (ModelRunner.kv_save /
kv_load / kv_drop are broadcast like graph calls).

State checkpoints (engine/state_pool.py): GLM-5.3-Flash's linear-attention layers resume a prefix
only where its recurrent state was checkpointed, so the host cache has a second part for them. A
checkpoint goes to host memory when its node is evicted, and also when only the checkpoint row is
taken back (RadixCache.evict_checkpoints, including one on a running request's path, and
free_dominated). It is keyed by the chain hash of the page it ends on (SGLang v0.5.21
mem_cache/pool_host/mamba.py keeps host copies of mamba states beside the KV the same way), and a
restore brings back the pages up to the deepest host checkpoint the prompt reaches, then that
checkpoint (Scheduler._restore).

Batched moves (EngineConfig.host_cache_batch): a restore's pages and an eviction's pages move in one
broadcast each, every cache copied once per run of consecutive device pages (ModelRunner.kv_load_many
/ kv_save_many), instead of one broadcast and one copy per cache per page. The same bytes land in the
same places. The per-page calls, not the bytes, were the cost of a restore at the KV wall.

Save-once (EngineConfig.host_cache_save_once, engine/host_arena.py): the ranks that hold identical
bytes keep one shared copy of a page instead of one each. The host cache then stamps every save with
the page's chain hash and has every restore check it, and it does not evict a slot a restore may
still be reading on a slower rank (ModelRunner.so_busy), nor hand one out again after clear().
"""

from __future__ import annotations

import itertools
from collections import OrderedDict


def page_run_count(pages: list[int]) -> int:
    """Runs of consecutive device pages in a batch: the copies per cache it costs (ModelRunner page_runs cuts them)."""
    s = sorted(pages)
    return len(s) - sum(b == a + 1 for a, b in itertools.pairwise(s))


class HostTier:
    def __init__(self, runner, page_size: int, capacity_pages: int, capacity_ckpts: int, batch: bool = False):
        """runner: the device copies, kv_save / kv_load / kv_drop (slot, page), with batch kv_save_many /
        kv_load_many (slots, pages), and state_save / state_load / state_drop (slot, row); capacities in host slots
        of one page or one state row. batch: a restore's and an eviction's pages in one call each."""
        if capacity_pages < 1 or capacity_ckpts < 1:
            raise ValueError("the host cache needs room for at least one page and one state checkpoint")
        self.runner, self.ps, self.capacity, self.batch = runner, page_size, capacity_pages, batch
        self.entries: OrderedDict[int, int] = OrderedDict()  # page hash -> host slot
        # host slot -> the page hash it holds (filled when the page is offloaded): the tag a save-once save stamps
        self.slot_hash: dict[int, int] = {}
        # host slot -> the page hash a lookup() asked for it by (pinned slots): the tag a save-once restore expects
        self.looked: dict[int, int] = {}
        self.free_slots = list(range(capacity_pages - 1, -1, -1))
        self.pinned: set[int] = set()
        # save-once: slots clear() took back while a restore may still be reading them on a slower rank (so_busy), kept
        # out of free_slots until that copy is done (_slot returns them)
        self.held: set[int] = set()
        self.saved = self.restored = self.dropped = 0
        # Recurrent-state checkpoints: hash of the page a checkpoint ends on -> host slot.
        self.ckpt_capacity = capacity_ckpts
        self.ckpts: OrderedDict[int, int] = OrderedDict()
        self.free_ckpt_slots = list(range(capacity_ckpts - 1, -1, -1))
        self.ckpts_saved = self.ckpts_restored = 0
        self.restores = self.restore_runs = self.save_runs = 0  # what the moves cost: calls and page runs copied

    def hashes(self, ids: list[int], n_pages: int) -> list[int]:
        out, h, ps = [], 0, self.ps
        for j in range(n_pages):
            h = hash((h, tuple(ids[j * ps : (j + 1) * ps])))
            out.append(h)
        return out

    def offload(self, path: list[int], first_page: int, pages: list[int], ckpt: int | None = None) -> None:
        """Keep the KV of `pages`, which hold the path's pages first_page, first_page + 1, ...,
        and `ckpt`, the state checkpoint row after the whole path, if the node had one."""
        hs = self.hashes(path, first_page + len(pages))[first_page:]
        pend: list[tuple[int, int]] = []  # batched saves (slot, page) not sent yet
        for h, page in zip(hs, pages):
            if h in self.entries:  # write-through: still held from an earlier eviction
                self.entries.move_to_end(h)
                continue
            slot = self._slot(pend)
            if slot is None:
                self._flush(pend)
                return
            if self.batch:
                pend.append((slot, page))
            else:
                self.runner.kv_save(slot, page)
                self.save_runs += 1
            self.entries[h] = slot
            self.slot_hash[slot] = h
            self.saved += 1
        self._flush(pend)
        if ckpt is not None:
            self.offload_ckpt(path, ckpt)

    def offload_ckpt(self, path: list[int], row: int) -> None:
        """Keep the state checkpoint `row`, the state after the page-aligned `path`."""
        if not path:
            return
        self.offload_ckpt_hash(self.hashes(path, len(path) // self.ps)[-1], row)

    def offload_ckpt_hash(self, h: int, row: int) -> None:
        """offload_ckpt by the path's last page hash (hashes(path)[-1]): the radix tree keeps it per node
        (RadixCache.path_hash), so a checkpoint eviction needn't rebuild and rehash the whole path."""
        if h in self.ckpts:
            self.ckpts.move_to_end(h)
            return
        if self.free_ckpt_slots:
            slot = self.free_ckpt_slots.pop()
        else:  # least recently used first
            _, slot = self.ckpts.popitem(last=False)
            self.runner.state_drop(slot)
        self.runner.state_save(slot, row)
        self.ckpts[h] = slot
        self.ckpts_saved += 1

    def find_ckpt(self, ids: list[int], lo: int, hi: int, chain: list[int] | None = None) -> tuple[int, int] | None:
        """The deepest page count j in [lo, hi] whose prefix ids[: j * page_size] has a host
        checkpoint, and its slot. chain: ids' page hashes as hashes() gives them (scheduler.page_chain), used
        when it covers the hi pages."""
        if hi < max(lo, 1):
            return None
        hs = chain if chain is not None and len(chain) >= hi else self.hashes(ids, hi)
        for j in range(hi, max(lo, 1) - 1, -1):
            slot = self.ckpts.get(hs[j - 1])
            if slot is not None:
                self.ckpts.move_to_end(hs[j - 1])
                return j, slot
        return None

    def load_ckpt(self, slot: int, row: int) -> None:
        self.runner.state_load(slot, row)
        self.ckpts_restored += 1

    def _flush(self, pend: list[tuple[int, int]]) -> None:
        """Send the batched saves collected so far, in one call. offload() maps each page's hash to its slot before the
        call: if the call raises, those slots may hold some, all or none of their pages, so their entries are taken
        back and the slots freed (a single-rank engine serves on after a failed step, and a later lookup must not
        return a slot with no payload)."""
        if pend:
            slots = [s for s, _ in pend]
            try:
                self.runner.kv_save_many(slots, [p for _, p in pend], **self._tags(slots))
            except BaseException:
                for s in slots:
                    h = self.slot_hash.pop(s, None)
                    if h is not None and self.entries.get(h) == s:
                        del self.entries[h]
                    self.free_slots.append(s)
                self.saved -= len(slots)
                pend.clear()
                raise
            self.save_runs += page_run_count([p for _, p in pend])
            pend.clear()

    def _tags(self, slots: list[int], restore: bool = False) -> dict:
        """Under save-once only (the runner's arenas are up): the tag of each slot's page, for kv_save_many to stamp and
        kv_load_many to check. A save stamps the hash the page was offloaded under (its radix path); a restore expects
        the hash the LOOKUP asked the slot by (the prompt's own chain), so a host cache whose hash -> slot map drifted
        from what the slots hold disagrees with the arena instead of agreeing with itself."""
        active = getattr(self.runner, "so_active", None)
        if active is None or not active():
            return {}
        from .host_arena import tag_of

        want = self.looked if restore else self.slot_hash
        return {"tags": [tag_of(want[s] if s in want else self.slot_hash[s]) for s in slots]}

    def _slot(self, pend: list[tuple[int, int]] | None = None) -> int | None:
        busy = getattr(self.runner, "so_busy", None)  # save-once: a slot a restore may still be reading is not evicted
        if self.held:
            done = {s for s in self.held if busy is None or not busy(s)}
            self.held -= done
            self.free_slots.extend(sorted(done, reverse=True))
        if self.free_slots:
            return self.free_slots.pop()
        for h, slot in self.entries.items():  # least recently used first
            if slot not in self.pinned and not (busy is not None and busy(slot)):
                if pend:  # the pending saves go before the drop, as they would one page at a time
                    self._flush(pend)  # first: if it raises, the entry it would evict still holds its page
                del self.entries[h]
                self.slot_hash.pop(slot, None)  # after the flush: a pending save of this slot is stamped with its tag
                self.runner.kv_drop(slot)
                self.dropped += 1
                return slot
        return None

    def lookup(self, ids: list[int], start_page: int, limit: int, chain: list[int] | None = None) -> list[int]:
        """Host slots of the pages start_page .. that continue a device match, up to page
        `limit` (exclusive), pinned until load() or unpin(). chain: as find_ckpt's."""
        if start_page >= limit:
            return []
        slots = []
        hs = chain[:limit] if chain is not None and len(chain) >= limit else self.hashes(ids, limit)
        for h in hs[start_page:]:
            slot = self.entries.get(h)
            if slot is None:
                break
            self.entries.move_to_end(h)
            slots.append(slot)
            self.looked[slot] = h
        self.pinned.update(slots)
        return slots

    def load(self, slots: list[int], pages: list[int]) -> None:
        if self.batch:
            if slots:
                self.runner.kv_load_many(
                    list(slots), list(pages[: len(slots)]), **self._tags(list(slots), restore=True)
                )
                self.restore_runs += page_run_count(pages[: len(slots)])
        else:
            for slot, page in zip(slots, pages):
                self.runner.kv_load(slot, page)
            self.restore_runs += len(slots)
        self.restores += bool(slots)
        self.restored += len(slots)
        self.unpin(slots)

    def clear(self) -> None:
        busy = getattr(self.runner, "so_busy", None)
        for slot in self.entries.values():
            self.runner.kv_drop(slot)
        # save-once: a slot a restore may still be reading on a slower rank is not handed out again until that copy is
        # done, as _slot's eviction never takes one: the next save would stamp a newer generation under the copy
        held = {*self.entries.values(), *self.held}
        self.held = {s for s in held if busy is not None and busy(s)}
        self.entries.clear()
        self.slot_hash.clear()
        self.looked.clear()
        self.pinned.clear()
        self.free_slots = [s for s in range(self.capacity - 1, -1, -1) if s not in self.held]
        for slot in self.ckpts.values():
            self.runner.state_drop(slot)
        self.ckpts.clear()
        self.free_ckpt_slots = list(range(self.ckpt_capacity - 1, -1, -1))

    def unpin(self, slots: list[int]) -> None:
        self.pinned.difference_update(slots)
        for s in slots:
            self.looked.pop(s, None)
