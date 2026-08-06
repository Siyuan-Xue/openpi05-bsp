# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires only the LIBERO submodule for this benchmark:

```bash
git submodule update --init --recursive third_party/libero
```

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

export CODE_SHA="$(git rev-parse HEAD)"
export LIBERO_DATASET_REVISION=v2.1
export POLICY_CONTAINER_DIGEST="$(docker image inspect openpi_server --format '{{.Id}}')"
```

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
--args.output-dir /experiments/eval/official-pi05-libero-h10-smoke \
--args.config-name pi05_libero \
--args.checkpoint-step 30000 \
--args.code-sha ${CODE_SHA} \
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
--args.output-dir /experiments/eval/baseline-h16-libero-10-smoke \
--args.config-name pi05_libero_baseline_h16 \
--args.checkpoint-step 30000 \
--args.code-sha ${CODE_SHA} \
--args.dataset-revision ${LIBERO_DATASET_REVISION} \
--args.norm-hash ${BASELINE_NORM_HASH} \
--args.checkpoint ${BASELINE_CHECKPOINT} \
--args.container-digest ${POLICY_CONTAINER_DIGEST} \
--args.train-seed 42 \
--args.eval-seed 42"

docker compose -f examples/libero/compose.yml up
```

If EGL fails after the NVIDIA/EGL installation has been checked, GLX/X11 is a
troubleshooting fallback only. It requires a host X server and an explicit
one-off socket mount; these are deliberately absent from the normal Compose
configuration:

```bash
sudo xhost +local:docker
docker compose -f examples/libero/compose.yml up -d openpi_server
docker compose -f examples/libero/compose.yml run --rm \
  -e DISPLAY="$DISPLAY" -e MUJOCO_GL=glx -e PYOPENGL_PLATFORM=glx \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro runtime
```

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

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
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
