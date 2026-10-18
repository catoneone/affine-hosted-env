"""Infrastructure: everything that is the same for every environment, and nothing that is specific to one.

- gateway/      public HTTP edge: hotkey signatures, replay protection, rate limits, routing
- coordinator/  quotas, task assignment, global budget, nodes, capacity planning (SQLite)
- executor/     the rollout loop: sessions, token rendering, forwarding tool calls to the environment, grading
                through the environment, signing receipts
- providers/    sandbox/ (one container per rollout: k8s, docker, local) and machine/ (boat, targon, static)
- auth.py chain.py config.py common.py
- receipt.py verify.py   receipt signing / verification, group verification (what validators need)
- render.py toolcall.py  chat-template rendering and tool-call parsing (the model side of the protocol)
- client.py              the reference miner

The boundary: infra talks to environments only through the interface in hosted_env.envs (the registry and
envs.base). It never imports a concrete environment such as hosted_env.envs.swe.
"""
