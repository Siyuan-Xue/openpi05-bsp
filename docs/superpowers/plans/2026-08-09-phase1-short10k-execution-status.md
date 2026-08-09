# Phase-one Short-10k Execution Status

## Status

- State: **paused by user before implementation**
- Recorded: 2026-08-09 12:02 UTC
- Approved target protocol: checkpoints at `0`, `1_000`, `2_000`, `5_000`, and `10_000` optimizer steps
- No short-10k feature worktree has been created.
- No training/config/report code has been changed for this protocol yet.
- No replacement 10k training process has been started.

## Repository state at pause

- Local branch: `main`
- Local HEAD: `3e286b6` (`docs: plan phase-one short 10k migration`)
- Remote `origin/main`: `196651804f21d25f4b92f0f0d67801e42b140089`
- Local `main` is ahead of `origin/main` by two documentation commits:
  - `ce111f9 docs: design phase-one short 10k protocol`
  - `3e286b6 docs: plan phase-one short 10k migration`
- Approved design: `docs/superpowers/specs/2026-08-09-phase1-short10k-protocol-design.md`
- Execution plan: `docs/superpowers/plans/2026-08-09-phase1-short10k-protocol.md`

## Stopped server training

- Server code SHA: `196651804f21d25f4b92f0f0d67801e42b140089`
- Config: `pi05_libero_baseline_lora_h16`
- Experiment: `phase1-seed42-baseline`
- PID: `1718659`
- Last observed progress: optimizer step `212 / 30_000`
- Existing checkpoint directories: `0` only
- Log error matches: `0`
- Termination: verified experiment identity, then graceful `SIGTERM`
- Process state after termination: stopped
- GPU compute-process count after termination: `0`
- Existing log and checkpoint were preserved; nothing was deleted or overwritten.

Server audit record:

```text
/root/openpi-bsp-work/experiments/logs/protocol-transition-30k-to-10k-20260809T120259Z-2211334.txt
sha256: 36f9ba560adbad8caf946ff30f30a60a7ef57d4a24864ff2820d22e69d7aa7a6
```

## Resume point

When the user asks to continue:

1. Reconfirm the server has no GPU compute process and that the stopped 30k run remains untouched.
2. Create `.worktrees/phase1-short10k` on branch `feat/phase1-short10k` from the current local `main`.
3. Follow `docs/superpowers/plans/2026-08-09-phase1-short10k-protocol.md` using test-first changes.
4. Change only the formal phase-one protocol to `10_000` steps with permanent milestones `(0, 1_000, 2_000, 5_000, 10_000)`; preserve references to the official OpenPI `pi05_libero@30k` calibration checkpoint.
5. Run the targeted config, planning, report, CLI, and server-contract tests.
6. Merge the verified worktree branch into `main`, push, then pull on the server (retry Git with HTTP/1.1 when necessary).
7. Run server-side verification and launch a new, uniquely named baseline LoRA 10k experiment. Never resume or overwrite `phase1-seed42-baseline`.
8. Point the hourly monitor at the new PID, log, checkpoint root, and `10_000`-step denominator only after the new training passes its startup gates.

## Safety constraints retained

- Do not delete any existing checkpoint, log, dataset, sidecar, or norm-stat artifact.
- Do not run more than one GPU job at a time.
- Do not write anywhere under `/mnt/data` except `/mnt/data/siyuanxue/`.
- Do not write Orbax checkpoints directly to the `ossfs2` object-storage mount.
- Do not install or modify the server system Python, CUDA, driver, or Docker stack.
