# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires only the LIBERO submodule for this benchmark:

```bash
git submodule update --init --recursive third_party/libero
```

## Host-only dual-Python setup

Run the policy server with the repository's Python 3.11 environment and the
LIBERO simulator client with Python 3.8. Keep their virtual environments
separate. The evaluator's `container-digest` field records a reproducible host
runtime fingerprint here; it is not an image identifier.

First, from the repository root, prepare both environments and audit identity:

```bash
export EXPERIMENTS_DIR=/absolute/path/to/experiments
export CODE_SHA="$(git rev-parse HEAD)"
export LIBERO_DATASET_REVISION=v2.0

# Server runtime: Python 3.11.
uv sync --python 3.11

# Simulator runtime: Python 3.8.
uv venv --python 3.8 examples/libero/.venv
examples/libero/.venv/bin/python -m pip install -r examples/libero/requirements.txt -r third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
examples/libero/.venv/bin/python -m pip install -e packages/openpi-client -e third_party/libero

# The digest binds both interpreters, the LIBERO checkout, and this host setup.
export HOST_RUNTIME_DIGEST="sha256:$(
  {
    printf 'server='; uv run --no-sync python --version
    printf 'simulator='; examples/libero/.venv/bin/python --version
    printf 'libero='; git -C third_party/libero rev-parse HEAD
  } | shasum -a 256 | awk '{print $1}'
)"
```

The official `pi05_libero` checkpoint emits horizon 10. In a server terminal,
run the default policy:

```bash
uv run --no-sync scripts/serve_policy.py --env LIBERO
```

In a simulator terminal, activate the Python 3.8 environment and run a true
one-task, one-rollout smoke on task 0 of `libero_spatial`. Replace the norm
placeholder with the verified SHA-256 of the official norm stats before
running:

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$PWD/third_party/libero"
export OFFICIAL_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero/
export OFFICIAL_NORM_HASH=REPLACE_WITH_VERIFIED_PI05_LIBERO_NORM_SHA256
export CLIENT_ARGS="\
--args.task-suite-name libero_spatial \
--args.task-ids 0 \
--args.policy-variant baseline \
--args.expected-action-horizon 10 \
--args.num-trials-per-task 1 \
--args.output-dir ${EXPERIMENTS_DIR}/eval/official-pi05-libero-h10-smoke \
--args.config-name pi05_libero \
--args.checkpoint-step 30000 \
--args.code-sha ${CODE_SHA} \
--args.dataset-revision ${LIBERO_DATASET_REVISION} \
--args.norm-hash ${OFFICIAL_NORM_HASH} \
--args.checkpoint ${OFFICIAL_CHECKPOINT} \
--args.container-digest ${HOST_RUNTIME_DIGEST} \
--args.train-seed 42 \
--args.eval-seed 42"

python examples/libero/main.py $CLIENT_ARGS
```

For a trained phase-one baseline, use the strict horizon-16 protocol. Start a
server terminal with the selected checkpoint, then run one rollout for each of
the ten tasks in `libero_10` from the simulator terminal:

```bash
export BASELINE_CHECKPOINT="${EXPERIMENTS_DIR}/checkpoints/pi05_libero_baseline_h16/REPLACE_WITH_RUN_NAME/30000"
uv run --no-sync scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi05_libero_baseline_h16 --policy.dir "${BASELINE_CHECKPOINT}"
```

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$PWD/third_party/libero"
export BASELINE_NORM_HASH=REPLACE_WITH_VERIFIED_BASELINE_NORM_SHA256
export CLIENT_ARGS="\
--args.task-suite-name libero_10 \
--args.policy-variant baseline \
--args.expected-action-horizon 16 \
--args.num-trials-per-task 1 \
--args.output-dir ${EXPERIMENTS_DIR}/eval/baseline-h16-libero-10-smoke \
--args.config-name pi05_libero_baseline_h16 \
--args.checkpoint-step 30000 \
--args.code-sha ${CODE_SHA} \
--args.dataset-revision ${LIBERO_DATASET_REVISION} \
--args.norm-hash ${BASELINE_NORM_HASH} \
--args.checkpoint ${BASELINE_CHECKPOINT} \
--args.container-digest ${HOST_RUNTIME_DIGEST} \
--args.train-seed 42 \
--args.eval-seed 42"

python examples/libero/main.py $CLIENT_ARGS
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
