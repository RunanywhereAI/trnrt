"""Shared host arenas for the save-once host cache (EngineConfig.host_cache_save_once).

Under context-parallel DSA at degree C below the attention TP A, the attention group is R = A / C row groups of C
consecutive ranks, and every row group keeps a full copy of each sequence's cache: the ranks of one CP index hold
identical bytes for every page. The write-through host cache (engine/host_cache.py) copies a page's slice into every
rank's own heap, R copies of the same bytes. Here a page's slice is copied ONCE, into the shared-memory arena of the
row group that OWNS its host slot (owner = slot mod R), and a restore reads it from there on all R ranks: R times the
pages in the same host memory, and each rank copies only 1 / R of the pages on a save.

    Arena   one named shared-memory segment per rank (the slots its row group owns): a header, then per slot one
            64-bit GENERATION cell and one 64-bit TAG cell (the chain hash of the page the slot holds, stamped by the
            owner and checked by every restore), then the slot payloads.
    Board   one segment, one 64-bit cell per rank: the number of the last host-cache command the rank FINISHED (the
            calls' shared-memory channel acknowledges receipt only).
    Layout  where each paged cache's rows of one page sit inside a slot payload.

The two ordering hazards an acknowledge-on-receipt channel leaves:
  1. a reader reaches "load slot S" before the owner finished "save S": the load carries the generation the save
     stamped, and the reader spins on the slot's cell until it equals it (wait_gen; a timeout raises, it never reads
     stale bytes). The save precedes the load in the broadcast order, and a rank reaches the load only after finishing
     every earlier command, so the wait cannot deadlock. The cell is a seqlock: the owner writes BUSY | gen before the
     payload and gen after it, and the reader checks the cell again after its copy (check_gen), so a payload rewritten
     under the copy raises too;
  2. a slot freed and saved again while a slow rank still copies from it: rank 0 keeps a slot busy until
     Board.min_done() has reached the sequence number of the last load that used it (ModelRunner.so_busy), so the
     host cache's LRU skips it: pinned through the copy, not only through the broadcast.

Names are short (macOS caps a POSIX shared-memory name at 31 characters) and carry a tag unique to the engine; stale
segments of a dead engine with the same tag are unlinked at start (cleanup_stale), and a rank unlinks its own at close,
at exit and on SIGTERM. The arenas' worst case is checked against the free space of /dev/shm before any is created,
less what the arenas of other engines on the host may still take (check_capacity). It is a check, not a reservation: two
engines that run it at the same moment can both pass (`python -m trnrt serve` starts its engines one after another).
"""

from __future__ import annotations

import atexit
import contextlib
import mmap
import os
import shutil
import time
from typing import Any

import numpy as np
import torch

PREFIX = "trnrthkv"
ALIGN = 64
HDR = 64
SPIN_TIMEOUT_S = 120.0
# A slot's generation cell holds BUSY | gen while its owner writes that generation's payload (HostStore.begin), gen once
# it is complete (commit). A reader waits for the cell to EQUAL the generation it expects and checks it again after its
# copy (check_gen): a seqlock, so a payload rewritten under a copy is an error, never a silent torn read.
BUSY = 1 << 62
# A slot's tag cell holds the 64-bit image of the chain hash of the page it holds (HostTier.hashes: a page's hash covers
# its own tokens and its parent's hash), 0 = untagged. The owner stamps it before the generation cell commits, so a
# reader that sees the generation sees the tag; a restore compares it with the hash the host cache looked the slot up
# by, so a slot index that drifted between the save and the load (bookkeeping the generation cannot see, since both
# sides read one counter) is a loud error.
NOTAG = 0
MASK = (1 << 64) - 1
SHM_DIR = "/dev/shm"  # noqa: S108  # where POSIX shared memory lives on Linux: the segments' names, not temp files


def tag_of(h: int) -> int:
    """The tag cell's value for a page's chain hash (Python's tuple hash is a signed 64-bit int; 0 means untagged)."""
    return (h & MASK) or 1


def _round(n: int, a: int) -> int:
    return -(-n // a) * a


def arena_name(tag: str, rank: int) -> str:
    return f"{PREFIX}{tag}_a{rank}"


def board_name(tag: str) -> str:
    return f"{PREFIX}{tag}_b"


def peers(rank: int, world: int, attn_tp: int, cp: int) -> list[int]:
    """The ranks that hold this rank's CP slice of every page, indexed by row group g = 0 .. R - 1: the ranks of its
    attention group with its CP index. peers(r)[row_group(r)] is r itself."""
    if world % attn_tp or attn_tp % cp:
        raise ValueError(f"world {world}, attention TP {attn_tp}, CP {cp} do not nest")
    base = (rank // attn_tp) * attn_tp
    c = (rank % attn_tp) % cp
    return [base + g * cp + c for g in range(attn_tp // cp)]


def row_group(rank: int, attn_tp: int, cp: int) -> int:
    return (rank % attn_tp) // cp


def _nbytes(lps: int, shape: tuple, dtype: torch.dtype) -> int:
    return lps * int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()


class Layout:
    """Where each paged cache's rows of ONE page sit in a slot payload: a 64-byte aligned region per cache,
    [lps, *row_shape] of the cache's dtype."""

    def __init__(self, caches: list[torch.Tensor], lps: int):
        self.lps = lps
        self.dtypes = [c.dtype for c in caches]
        self.shapes = [tuple(c.shape[1:]) for c in caches]
        self.offsets: list[int] = []
        off = 0
        for dt, shp in zip(self.dtypes, self.shapes):
            self.offsets.append(off)
            off += _round(_nbytes(lps, shp, dt), ALIGN)
        self.slot_bytes = _round(off, ALIGN)

    def arena_views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        """flat: an arena's payloads as a [capacity, slot_bytes] uint8 tensor. One persistent view per cache over ALL
        slots, [capacity, lps, *shape] in the cache's dtype, sharing the memory: a slot's views are then one index per
        cache instead of a slice, a view and a reshape per cache per slot (measured 19.7 us against 1.9 us a slot)."""
        out = []
        for off, dt, shp in zip(self.offsets, self.dtypes, self.shapes):
            n = _nbytes(self.lps, shp, dt)
            out.append(flat[:, off : off + n].view(dt).reshape(flat.shape[0], self.lps, *shp))
        return out

    def views(self, payload: torch.Tensor) -> list[torch.Tensor]:
        """payload: a uint8 tensor of slot_bytes (one slot). The per-cache views [lps, *shape] in each cache's dtype,
        sharing its memory."""
        out = []
        for off, dt, shp in zip(self.offsets, self.dtypes, self.shapes):
            n = _nbytes(self.lps, shp, dt)
            out.append(payload[off : off + n].view(dt).reshape(self.lps, *shp))
        return out


class _Mapped:
    """A segment mapped by a reader: the buffer and its close. No unlink: the creator owns the name."""

    def __init__(self, name: str, mm: mmap.mmap):
        self.name, self._mm = name, mm
        self.buf = memoryview(mm)
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self.buf.release()  # BufferError while a numpy or torch view of it is alive: the mapping stays until exit
        self._mm.close()

    def unlink(self) -> None:
        raise RuntimeError(f"{self.name} is not ours to unlink")


class _Seg:
    """A named shared-memory segment. The creator registers and unlinks it; a reader never touches the resource
    tracker."""

    def __init__(self, shm, owner: bool):
        self.shm, self.owner, self.closed = shm, owner, False

    @staticmethod
    def create(name: str, size: int) -> _Seg:
        from multiprocessing import shared_memory

        return _Seg(shared_memory.SharedMemory(name=name, create=True, size=size), True)

    @staticmethod
    def attach(name: str, timeout: float = 30.0) -> _Seg:
        """Map a segment another rank created. Its name is derived, not handed over after creation, so a reader can
        reach a name its creator has made but not yet sized: multiprocessing.shared_memory fails there (a zero-length
        mmap) and, on its failure path, UNLINKS the name from under the creator (measured on macOS: one rank failed and
        every other waited for a board that no longer existed). So the object is opened here, mapped only once it has a
        size, and never unlinked."""
        import _posixshmem  # CPython's POSIX shared-memory module

        end = time.time() + timeout
        last = "does not exist"
        while True:
            try:
                fd = _posixshmem.shm_open("/" + name, os.O_RDWR, mode=0o600)
            except FileNotFoundError:
                fd = None
            if fd is not None:
                try:
                    size = os.fstat(fd).st_size
                    if size > 0:
                        return _Seg(_Mapped(name, mmap.mmap(fd, size)), False)
                    last = "exists but has no size yet"
                finally:
                    os.close(fd)
            if time.time() > end:
                raise TimeoutError(f"shared segment {name} {last} after {timeout:.0f} s: its owner never created it")
            time.sleep(0.005)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(BufferError):  # a view is still alive in this process; the OS frees it at exit
            self.shm.close()
        if self.owner:
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()


class Arena:
    """One rank's segment of host slots: [ctl: capacity, slot_bytes][gen[capacity]][tag[capacity]][payloads,
    4096-aligned]. gen[i] is the generation of the page in slot i (0: empty), tag[i] its chain hash (tag_of)."""

    def __init__(self, seg: _Seg):
        self.seg = seg
        buf = seg.shm.buf
        ctl = np.ndarray((2,), dtype=np.uint64, buffer=buf, offset=0)
        self.capacity, self.slot_bytes = int(ctl[0]), int(ctl[1])
        self.gen = np.ndarray((self.capacity,), dtype=np.uint64, buffer=buf, offset=HDR)
        self.tag = np.ndarray((self.capacity,), dtype=np.uint64, buffer=buf, offset=HDR + 8 * self.capacity)
        self.data_off = _round(HDR + 16 * self.capacity, 4096)
        n = self.capacity * self.slot_bytes
        self.bytes = (
            torch.frombuffer(buf, dtype=torch.uint8, offset=self.data_off, count=n)
            if n
            else torch.empty(0, dtype=torch.uint8)
        )
        self.slot_views: list[torch.Tensor] = []  # HostStore.arena_of fills them (they need the layout)

    @staticmethod
    def total_bytes(capacity: int, slot_bytes: int) -> int:
        return _round(HDR + 16 * capacity, 4096) + capacity * slot_bytes

    @classmethod
    def create(cls, name: str, capacity: int, slot_bytes: int) -> Arena:
        seg = _Seg.create(name, cls.total_bytes(capacity, slot_bytes))
        np.ndarray((2 * capacity,), dtype=np.uint64, buffer=seg.shm.buf, offset=HDR)[:] = 0  # gen and tag cells
        ctl = np.ndarray((2,), dtype=np.uint64, buffer=seg.shm.buf, offset=0)
        ctl[1] = slot_bytes
        ctl[0] = capacity  # LAST: a reader takes capacity 0 as "not published yet" (attach)
        return cls(seg)

    @classmethod
    def attach(cls, name: str, timeout: float = 30.0) -> Arena:
        end = time.time() + timeout
        while True:
            seg = _Seg.attach(name, max(0.0, end - time.time()))
            ctl = np.ndarray((2,), dtype=np.uint64, buffer=seg.shm.buf, offset=0)
            ready = int(ctl[0]) > 0 and int(ctl[1]) > 0
            del ctl
            if ready:
                return cls(seg)
            seg.close()  # sized, but its header is not written yet: look again
            if time.time() > end:
                raise TimeoutError(
                    f"shared segment {name} exists but its owner never published its header after {timeout:.0f} s"
                )
            time.sleep(0.005)

    def payload(self, i: int) -> torch.Tensor:
        return self.bytes[i * self.slot_bytes : (i + 1) * self.slot_bytes]

    def close(self) -> None:
        # drop this arena's views of the buffer first, so the segment's own close can release it
        del self.gen, self.tag, self.bytes
        self.slot_views = []
        self.seg.close()


class Board:
    """done[rank]: the sequence number of the last host-cache command the rank finished. One writer per cell."""

    def __init__(self, seg: _Seg, world: int):
        self.seg, self.world = seg, world
        self.done = np.ndarray((world,), dtype=np.uint64, buffer=seg.shm.buf, offset=0)

    @staticmethod
    def total_bytes(world: int) -> int:
        return _round(8 * world, 64)

    @classmethod
    def create(cls, name: str, world: int) -> Board:
        b = cls(_Seg.create(name, cls.total_bytes(world)), world)
        b.done[:] = 0
        return b

    @classmethod
    def attach(cls, name: str, world: int, timeout: float = 30.0) -> Board:
        return cls(_Seg.attach(name, timeout), world)

    def mark(self, rank: int, seq: int) -> None:
        self.done[rank] = seq

    def min_done(self) -> int:
        return int(self.done.min())

    def close(self) -> None:
        del self.done
        self.seg.close()


def cleanup_stale(tag: str, names: list[str] | None = None) -> list[str]:
    """Unlink every segment of an engine with this tag that a dead engine left (shared memory survives a killed process
    and nothing reclaims it). `names`: explicit names (a platform without /dev/shm); else the files /dev/shm/<PREFIX>
    <tag>_*. Unlinked by name, never mapped: a segment of a rank killed between creating and sizing it has no size and
    cannot be mapped. Returns the names removed."""
    import _posixshmem  # CPython's POSIX shared-memory module

    if names is None:
        names = []
        if os.path.isdir(SHM_DIR):
            names = [f for f in os.listdir(SHM_DIR) if f.startswith(f"{PREFIX}{tag}_")]
    gone = []
    for n in names:
        try:
            _posixshmem.shm_unlink("/" + n)
        except FileNotFoundError:
            continue
        gone.append(n)
    return gone


def engine_bytes(world: int, per_arena: int, slot_bytes: int) -> int:
    """The shared memory an engine's save-once host cache takes once full: every rank's arena (its header, its generation
    and tag cells, its payloads) and the board."""
    return world * Arena.total_bytes(per_arena, slot_bytes) + Board.total_bytes(world)


def unbacked_bytes(path: str = SHM_DIR, skip_tag: str | None = None) -> int:
    """What the save-once segments in `path` (every engine's but skip_tag's) may still take from the tmpfs: each one's
    size less the bytes already backed. A segment is sized when it is made and backed page by page as its host cache
    fills, so the free space does not show what an engine that started earlier will go on to take."""
    n = 0
    for f in os.listdir(path):
        if f.startswith(PREFIX) and not (skip_tag is not None and f.startswith(f"{PREFIX}{skip_tag}_")):
            with contextlib.suppress(FileNotFoundError):  # unlinked since the listing
                st = os.stat(os.path.join(path, f))
                n += max(0, st.st_size - st.st_blocks * 512)
    return n


def check_capacity(total_bytes: int, path: str = SHM_DIR, margin: float = 0.0, tag: str | None = None) -> None:
    """Refuse (ValueError) when the arenas' worst case does not fit the tmpfs: the segments are committed lazily, but a
    leak or a second engine could leave less free than a full host cache needs, and a write into an unbacked page is a
    SIGBUS in the middle of a save. What other engines' segments (any tag but `tag`) may still take is not free. Skipped
    where there is no such tmpfs."""
    if not os.path.isdir(path):
        return
    # others before free: a segment another engine backs between the two reads then lowers free after it was counted
    # as unbacked, so the headroom can only be understated, never counted twice
    others = unbacked_bytes(path, skip_tag=tag)
    free = shutil.disk_usage(path).free
    if total_bytes * (1.0 + margin) > free - others:
        raise ValueError(
            f"the save-once host arenas need up to {total_bytes / 2**30:.1f} GiB of {path} and only "
            f"{(free - others) / 2**30:.1f} GiB is free ({free / 2**30:.1f} GiB unused, of which other engines' arenas "
            f"may still fill {others / 2**30:.1f} GiB; or a leaked segment?): lower --host-cache-gb or remove the "
            "stale segments"
        )


class HostStore:
    """A rank's view of its CP index's arenas: it owns the arena of its row group, attaches the other R - 1 row groups'
    on first use, and shares the board. Slot s lives at index s // R of the arena of row group s % R."""

    def __init__(
        self,
        tag: str,
        rank: int,
        world: int,
        attn_tp: int,
        cp: int,
        layout: Layout,
        capacity_slots: int,
        create: bool = True,
    ):
        self.tag, self.rank, self.world, self.layout = tag, rank, world, layout
        self.attn_tp, self.cp = attn_tp, cp
        self.peer = peers(rank, world, attn_tp, cp)
        self.R = len(self.peer)
        self.mine = row_group(rank, attn_tp, cp)
        self.capacity = capacity_slots  # total slots, over the R row groups
        self.per_arena = -(-capacity_slots // self.R)
        self.arenas: dict[int, Arena] = {}
        self.board: Board | None = None
        self._term: Any = None
        self._prev_term: Any = None
        if create:
            self.arenas[self.mine] = Arena.create(arena_name(tag, rank), self.per_arena, layout.slot_bytes)
            # A worker's store closes when the worker exits. Rank 0's closes with its engine (LLMEngine.close, after the
            # step thread stopped and the workers are gone, through the engine module's exit hook): an exit hook of its
            # own, registered after the server's, ran first at exit and unmapped the arena and the board under a step
            # thread that could still move pages to the host cache.
            if rank != 0:
                atexit.register(self.close)
            self._hook_sigterm()

    def owner(self, slot: int) -> int:
        return slot % self.R

    def arena_of(self, g: int, timeout: float = 30.0) -> Arena:
        a = self.arenas.get(g)
        if a is None:
            a = self.arenas[g] = Arena.attach(arena_name(self.tag, self.peer[g]), timeout)
        if not a.slot_views and a.capacity:
            a.slot_views = self.layout.arena_views(a.bytes.view(a.capacity, a.slot_bytes))
        return a

    def attach_board(self, create: bool, timeout: float = 30.0) -> Board:
        self.board = (
            Board.create(board_name(self.tag), self.world)
            if create
            else Board.attach(board_name(self.tag), self.world, timeout)
        )
        return self.board

    def views(self, slot: int) -> list[torch.Tensor]:
        """The per-cache views of a slot's payload, in its owner's arena (this rank's own, or a peer's)."""
        a = self.arena_of(self.owner(slot))
        i = slot // self.R
        return [v[i] for v in a.slot_views]

    def _cells(self, slot: int) -> tuple[np.ndarray, np.ndarray, int]:
        a = self.arena_of(self.owner(slot))
        return a.gen, a.tag, slot // self.R

    def _owned(self, slot: int) -> None:
        if self.owner(slot) != self.mine:
            raise RuntimeError(f"slot {slot} is owned by row group {self.owner(slot)}, not {self.mine}")

    def begin(self, slot: int, gen: int) -> None:
        """The owner marks the slot as being written, BEFORE the first payload byte (x86 orders stores with stores and
        loads with loads: a reader whose copy saw any new byte reads this marker, or a later value, in check_gen)."""
        self._owned(slot)
        cells, _, i = self._cells(slot)
        cells[i] = BUSY | gen

    def commit(self, slot: int, gen: int, tag: int = NOTAG) -> None:
        """The owner stamps the generation AFTER the payload is written (a reader that sees the cell sees the bytes),
        and the tag BEFORE it, so a reader that sees the generation sees that generation's tag (a slot saved untagged is
        stamped NOTAG, never the previous page's)."""
        self._owned(slot)
        cells, tags, i = self._cells(slot)
        tags[i] = tag
        cells[i] = gen

    def tag_at(self, slot: int) -> int:
        """The tag the owner stamped on the slot (valid once wait_gen returned for the slot's generation)."""
        _, tags, i = self._cells(slot)
        return int(tags[i])

    def check_tag(self, slot: int, want: int) -> None:
        """The slot holds the page whose tag is `want` (tag_of its chain hash); else a restore would give some other
        page's KV to the prefix: a loud error before any byte is copied."""
        got = self.tag_at(slot)
        if got != want:
            raise RuntimeError(
                f"host slot {slot}: its tag is {got:#018x} but the restore looked it up as {want:#018x}: the slot "
                "does not hold the page the host cache believes it does"
            )

    def check_gen(self, slot: int, gen: int) -> None:
        """After a reader's copy: the cell still holds exactly the generation the load expected, else the owner began
        rewriting the slot during the copy."""
        cells, _, i = self._cells(slot)
        g = int(cells[i])
        if g != gen:
            being = " (being rewritten)" if g & BUSY else ""
            raise RuntimeError(
                f"host slot {slot}: its cell holds {g & (BUSY - 1)}{being}, the load copied generation {gen}: the slot "
                "was rewritten during the copy"
            )

    def wait_gen(self, slot: int, gen: int, timeout: float = SPIN_TIMEOUT_S) -> None:
        cells, _, i = self._cells(slot)
        n, end = 0, time.time() + timeout
        while True:
            g = int(cells[i])
            if g == gen:
                return
            if (g & (BUSY - 1)) > gen:  # a newer generation, finished or being written
                raise RuntimeError(
                    f"host slot {slot}: generation {g & (BUSY - 1)} is already in the arena, a load expected {gen}: "
                    "the slot was reused under a pending load"
                )
            n += 1
            if n > 2000:
                time.sleep(5e-5 if n < 20000 else 1e-3)
                if time.time() > end:
                    raise TimeoutError(
                        f"host slot {slot}: generation {gen} never arrived (the cell holds {g}) after {timeout:.0f} s: "
                        "its owner's save did not run"
                    )

    def _hook_sigterm(self) -> None:
        """A rank stopped by SIGTERM (a worker the engine terminates, a server's stop) unlinks its segments, then dies
        as it would have: atexit does not run on a signal. Only from the main thread (signal.signal's rule); SIGKILL
        cannot be caught, and the next start's cleanup_stale takes those segments."""
        import signal
        import threading

        if threading.current_thread() is not threading.main_thread():
            return
        prev = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):
            self.close()
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_IGN:
                return
            else:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, on_term)
        self._term, self._prev_term = on_term, prev

    def _unhook_sigterm(self) -> None:
        if self._term is None:
            return
        import signal
        import threading

        if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) is self._term:
            signal.signal(signal.SIGTERM, self._prev_term if self._prev_term is not None else signal.SIG_DFL)
        self._term = None

    def close(self) -> None:
        self._unhook_sigterm()
        for a in list(self.arenas.values()):
            a.close()
        self.arenas.clear()
        if self.board is not None:
            self.board.close()
            self.board = None
