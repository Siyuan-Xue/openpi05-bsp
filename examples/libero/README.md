# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires only the LIBERO submodule for this benchmark:

```bash
git submodule update --init --recursive third_party/libero
```

## Evaluation clocks and schema v3 video artifacts

The evaluator keeps four independent clocks: the LeRobot dataset is indexed at
10 FPS, the source LIBERO demonstrations were collected from a 20 Hz
environment, evaluation dynamics run at exactly 20 Hz, and selected MP4 files
default to 40 FPS. Dataset and source rates are provenance; video synthesis
uses only control steps, the 20 Hz control rate, the selected video FPS, and
measured control stalls.

Use `--args.control-freq 20 --args.video-fps 40`. Add
`--args.video-show-inference-waits` to display synchronous inference stalls.
The switch changes only selected MP4/video-audit artifacts: it adds no
`env.step`, dummy action, or sleep and cannot change success. The expected
duration is `control_steps / 20 + included_control_stall_seconds`, with a
warning tolerance of one output frame (`1 / video_fps`). Encoder/readback
errors remain artifact errors. Current synchronous requests measure latency
and stall over the same interval; a future asynchronous scheduler may have
latency without a stall, and only measured stalls freeze video.

Formal comparisons accept schema v3 only. Schema-v2 outputs remain immutable
historical archives and must not be mixed with v3. Existing checkpoints need
no retraining, but every formal evaluation input must be rerun into a new,
empty output directory. Formal commands omit `--args.code-sha`: the evaluator
records the clean checkout HEAD automatically. Selected videos are audited in
`video_audit.jsonl`.

## Schema v5 paired random-latency experiment

The evaluator is `examples/libero/main_v5.py`. Its original formal experiment
compares plain baseline async, continuity-guided baseline RTC, and continuous
BSP async under the same deterministic paired `Normal(300 ms, 60 ms)` request
targets. It now also exposes `baseline_sync` and `bsp_spline_sync` as a separate
synchronous extension. The fixed theoretical scheduling budget is 400 ms
(eight 20 Hz ticks); empirical calibration remains audit-only. Fixed latency
targets are not exposed by the schema-v5 CLI.

`baseline_sync` blocks for a policy response, then executes the complete
16-action model chunk before requesting the next one. `bsp_spline_sync` blocks
for a continuous BSP curve and executes it from `t_min` through its closed
endpoint before requesting a replacement; because no old curve runs during the
blocking request, every installed curve has zero phase offset. Both modes use
the same paired random-latency worker, 20 Hz controller, 40 FPS video, and
persistent cumulative-wait overlay as the asynchronous modes. The initial
request is a full stall; later synchronous requests count only the portion that
overruns the next 20 Hz control deadline, so the overlay remains a control-wait
measurement rather than a copy of raw inference latency.

The BSP mode uses protocol `bsp_spline_async_phase_skip_speedup2_v2`: knots
retain their 10 Hz dataset-index origin, inference-time `speedup=2` advances
the continuous curve at 20 indices/s, and each completed 20 Hz `env.step()`
advances exactly one index. A background response skips the prefix consumed
since its request; a response whose computed phase is past `t_max` is audited
and replaced by a blocking replan from the latest observation. The legacy
eight-action response preview is validated for transport compatibility but is
not executed by this formal scheduler.

The v5 video overlay is one persistent `Cumulative inference wait: X.XX s`
line with no solid background. Only real action-underflow stalls freeze video;
hidden async latency does not. Selected videos are streamed to the encoder one
frame at a time.

Do not mix v3 and v5 artifacts. The original three-mode report stays frozen as
`compare_libero_latency_v5.py`. A separate exact two-input synchronous report
is available as `compare_libero_sync_v5.py`; the two output sets cannot collide.
The complete sampling, reporting, server-gate, and interpretation contract is in
[`docs/pi05_libero_latency_experiment_v5.md`](../../docs/pi05_libero_latency_experiment_v5.md).

## With Docker (recommended)

The default Compose path is headless EGL and does not require an X server or
`xhost`. Compose intentionally has no working-directory or home-directory
mount fallback: set every source below to an existing **absolute** host path.
The preflight is read-only and rejects missing, relative, nonexistent, root,
or non-directory paths; Compose is also configured not to create bind sources.

```bash
export BSP_REPO_DIR=/mnt/workspace/openpi-bsp/repo/openpi05-bsp
export BSP_EXPERIMENTS_DIR=/mnt/workspace/openpi-bsp/experiments
export BSP_OPENPI_CACHE_DIR=/mnt/workspace/openpi-bsp/cache/openpi
export BSP_JAX_CACHE_DIR=/mnt/workspace/openpi-bsp/cache/jax

python3 scripts/libero_compose_preflight.py
docker compose -f examples/libero/compose.yml config >/dev/null
docker compose -f examples/libero/compose.yml build

if ! EXPECTED_REPO_SHA="$(git -C "$BSP_REPO_DIR" rev-parse HEAD)"; then
  echo "STOP: unable to resolve the selected host checkout" >&2
  exit 2
fi
case "$EXPECTED_REPO_SHA" in
  *[!0-9a-f]*|'') echo "STOP: invalid host checkout SHA" >&2; exit 2 ;;
esac
test "${#EXPECTED_REPO_SHA}" -eq 40
if ! host_status="$(git -C "$BSP_REPO_DIR" status --porcelain --untracked-files=all)"; then
  echo "STOP: unable to inspect the selected host checkout" >&2
  exit 2
fi
test -z "$host_status"

docker compose -f examples/libero/compose.yml run --no-deps \
  --name libero-git-identity-preflight \
  -e EXPECTED_REPO_SHA="$EXPECTED_REPO_SHA" \
  --entrypoint /bin/bash runtime -ceu '
    if ! container_sha="$(git -C /app rev-parse HEAD)"; then
      echo "STOP: runtime cannot resolve evaluator SHA" >&2
      exit 2
    fi
    test "$container_sha" = "$EXPECTED_REPO_SHA"
    if ! container_status="$(git -C /app status --porcelain --untracked-files=all)"; then
      echo "STOP: runtime cannot inspect evaluator checkout" >&2
      exit 2
    fi
    test -z "$container_status"
    printf "runtime_evaluator_sha=%s\n" "$container_sha"
  '

export LIBERO_DATASET_REVISION=v2.0
export POLICY_CONTAINER_DIGEST="$(docker image inspect openpi_server --format '{{.Id}}')"
```

The runtime image installs Git explicitly. Compose applies
`GIT_OPTIONAL_LOCKS=0` and exact `/app` and pinned LIBERO-submodule
`safe.directory` entries so the
clean-status operation used by the evaluator succeeds against the read-only
source bind mount. The preflight above creates a retained audit container; a
name collision is a stop condition rather than permission to remove it.

The official `pi05_libero` checkpoint emits horizon 10. The following is a
true one-task, one-rollout EGL/connectivity smoke on task 0 of
`libero_spatial`, not the full acceptance evaluation. Replace the norm
placeholder with the verified SHA-256 of the official norm stats before
running:

```bash
export OFFICIAL_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero/
export OFFICIAL_NORM_HASH=REPLACE_WITH_VERIFIED_PI05_LIBERO_NORM_SHA256
export SERVER_ARGS="--env LIBERO"
export CLIENT_ARGS="\
--args.task-suite-name libero_spatial \
--args.task-ids 0 \
--args.policy-variant baseline \
--args.expected-action-horizon 10 \
--args.num-trials-per-task 1 \
--args.control-freq 20 \
--args.video-fps 40 \
--args.video-show-inference-waits \
--args.output-dir /experiments/eval/official-pi05-libero-h10-smoke \
--args.config-name pi05_libero \
--args.checkpoint-step 30000 \
--args.dataset-revision ${LIBERO_DATASET_REVISION} \
--args.norm-hash ${OFFICIAL_NORM_HASH} \
--args.checkpoint ${OFFICIAL_CHECKPOINT} \
--args.container-digest ${POLICY_CONTAINER_DIGEST} \
--args.train-seed 42 \
--args.eval-seed 42"

docker compose -f examples/libero/compose.yml up
```

For a trained phase-one baseline, use the strict horizon-16 protocol. This
example again runs one rollout for each of the 10 tasks in the selected suite;
the full 50-rollout acceptance commands belong in the server runbook:

```bash
export BASELINE_CHECKPOINT=/experiments/checkpoints/pi05_libero_baseline_h16/REPLACE_WITH_RUN_NAME/30000
export BASELINE_NORM_HASH=REPLACE_WITH_VERIFIED_BASELINE_NORM_SHA256
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_baseline_h16 --policy.dir ${BASELINE_CHECKPOINT}"
export CLIENT_ARGS="\
--args.task-suite-name libero_10 \
--args.policy-variant baseline \
--args.expected-action-horizon 16 \
--args.num-trials-per-task 1 \
--args.control-freq 20 \
--args.video-fps 40 \
--args.video-show-inference-waits \
--args.output-dir /experiments/eval/baseline-h16-libero-10-smoke \
--args.config-name pi05_libero_baseline_h16 \
--args.checkpoint-step 30000 \
--args.dataset-revision ${LIBERO_DATASET_REVISION} \
--args.norm-hash ${BASELINE_NORM_HASH} \
--args.checkpoint ${BASELINE_CHECKPOINT} \
--args.container-digest ${POLICY_CONTAINER_DIGEST} \
--args.train-seed 42 \
--args.eval-seed 42"

docker compose -f examples/libero/compose.yml up
```

For the phase-one BSP comparison, do not fall back to GLX/X11, change host X
server access controls, or auto-remove evaluation containers. If the EGL or
nested-Docker gate fails, stop and follow the isolated-environment route in the
[authoritative server runbook](../../docs/pi05_libero_bsp_phase1_server.md),
which preserves the audit artifacts and stops if host EGL libraries are absent.

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Export the same auditable CLIENT_ARGS shown above, then run the simulation.
python examples/libero/main.py $CLIENT_ARGS

# For the phase-one protocol, use the isolated EGL route from the server
# runbook. Stop and report the inventory if the host EGL gate fails.
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
