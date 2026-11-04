# Security policy

## Reporting a vulnerability

Report vulnerabilities privately, never in a public issue:

- **GitHub:** [open a private security advisory](https://github.com/RunanywhereAI/trnrt/security/advisories/new), or
- **Email:** [security@runanywhere.ai](mailto:security@runanywhere.ai).

Please include:

- what the issue is and what an attacker could do with it;
- the affected version or commit;
- steps to reproduce it, or a proof of concept;
- any suggested fix.

We acknowledge reports within 3 business days and keep you updated until the issue is fixed. Please give us a
reasonable time to release a fix before you disclose it publicly.

## Supported versions

Security fixes go to the latest release and to `main`.

## Deployment notes

TrnRT is a model server with **no authentication**:

- The router listens on every interface by default (`--host 0.0.0.0`). Put it behind your own proxy or gateway, or
  pass `--host 127.0.0.1`.
- Under `serve`, engines listen on `127.0.0.1` only, so their `/metrics` endpoint is not exposed beyond the host. A
  standalone `python -m trnrt engine` binds `--host` (default `0.0.0.0`).
- The tensor-parallel ranks talk over local TCP and shared memory. Keep ports 21000 and up closed to other hosts.
- The serving container runs with `--privileged` to reach the Neuron devices. Run it only on a host you control.
- Pin `--revision` when serving a Hugging Face repo id, so the weights and chat template cannot change under you.
