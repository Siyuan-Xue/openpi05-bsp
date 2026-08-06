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
Unset variables fail during Compose interpolation before either container is
started.

```bash
export BSP_REPO_DIR=/mnt/workspace/openpi-bsp/repo/openpi05-bsp
export BSP_EXPERIMENTS_DIR=/mnt/workspace/openpi-bsp/experiments
export BSP_OPENPI_CACHE_DIR=/mnt/workspace/openpi-bsp/cache/openpi
export BSP_JAX_CACHE_DIR=/mnt/workspace/openpi-bsp/cache/jax

export SERVER_ARGS="--env LIBERO"
export CLIENT_ARGS="--output-dir /experiments/eval/libero-smoke"

docker compose -f examples/libero/compose.yml config >/dev/null
docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# Checkpoints and evaluation artifacts live outside the repository checkout.
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_baseline_h16 --policy.dir /experiments/checkpoints/pi05_libero_baseline_h16/run/30000"

# To run the libero_10 task suite:
export CLIENT_ARGS="--task-suite-name libero_10 --output-dir /experiments/eval/baseline-30k-libero-10"
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

# Run the simulation
python examples/libero/main.py

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
