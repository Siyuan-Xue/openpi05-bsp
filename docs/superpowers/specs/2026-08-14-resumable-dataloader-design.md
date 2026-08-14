# Resumable LIBERO DataLoader Design

## Context

The phase-one BSP LoRA run was interrupted while the Alibaba Cloud DSW
instance saved an image. The runtime repository was frozen at
`2c098404a3cce0c86f0b863dcd8d3aeb18a55d94`. Its Orbax checkpoints preserve
model parameters and optimizer state, but the training loader has no persisted
position. A plain `--resume` therefore restores optimizer step `N` while
starting the seed-42 shuffle stream again at its first batch.

For the current configuration, effective batch 256 and micro-batch 64 produce
four loader batches per optimizer step. Resuming step 2,000 from the beginning
of the loader would repeat the first 8,000 micro-batches and make the BSP data
schedule differ from the uninterrupted baseline schedule.

## Goals

- Recover the exact next LIBERO micro-batch from a legacy phase-one checkpoint
  that has no loader metadata.
- Make later interruptions recoverable with one `--resume` command.
- Preserve compatibility with existing Orbax `params`, `train_state`, and
  `assets` items.
- Refuse recovery when dataset identity, seed, batch topology, or sampler
  protocol does not match the checkpoint.
- Leave uninterrupted training behavior and checkpoint retention unchanged.
- Keep the runtime fix isolated from `main` and the repository-slimming branch.

## Non-goals

- Supporting RLDS/DROID loader resume.
- Supporting multi-process data loading, which the current loader already
  rejects.
- Changing phase-one seed, effective batch, micro-batch, worker count, model,
  BSP sidecar, norm stats, or checkpoint milestones.
- Hiding the code transition in experiment manifests.

## Selected architecture

### 1. Versioned loader cursor

A small JSON document is stored inside each checkpoint's existing `assets`
item. It contains:

- format version;
- completed optimizer step;
- consumed loader-batch count;
- seed and shuffle flag;
- dataset length and dataset identity;
- global and local micro-batch sizes;
- accumulation steps and process count;
- drop-last behavior;
- sampler protocol identifier.

The consumed count is derived from completed optimizer steps and the validated
gradient-accumulation plan, not from DataLoader prefetch progress. This avoids
worker prefetch creating an off-by-one cursor.

The cursor is written by the existing assets callback. The Orbax item layout is
not changed, so checkpoints written before this feature remain loadable.

### 2. Resume planning before the first batch

The checkpoint manager identifies the latest complete step before the loader's
first batch is requested. Resume planning then selects one of two paths:

1. If a cursor exists, validate every identity and topology field and use its
   consumed loader-batch count.
2. If the legacy checkpoint has no cursor, derive the count as
   `restored_step * accumulation_steps` and mark the recovery as a legacy
   reconstruction in the log.

The loader is positioned before the batch used for image logging or gradient
computation is read. Step `N + 1` therefore starts with the first batch not
consumed by step `N`.

### 3. Replayable shuffle sampler

The JAX LeRobot path receives a replayable random batch sampler. Given the
same seed, dataset length, local batch size, drop-last setting, and absolute
loader-batch offset, it reconstructs the same shuffled index stream as the
current pinned PyTorch DataLoader without decoding discarded samples.

The sampler advances prior epochs at index level and skips only indices, not
images or transformed observations. Its compatibility contract is the exact
index sequence produced by the current `shuffle=True`, seeded PyTorch loader.

If exact legacy parity cannot be demonstrated for the pinned server PyTorch
version, the implementation must fail closed. It must not silently use a new
shuffle algorithm; the safe fallback is restarting the BSP run from step zero.

### 4. Dataset identity

The cursor binds to the dataset inputs that affect ordering or labels:

- repository ID and revision;
- LeRobot dataset fingerprint and episode-boundary identity already used by
  the BSP manifest when available;
- dataset length;
- BSP cache fingerprint for BSP configurations;
- action horizon and action-key layout.

Resume rejects missing or mismatched required fields for newly written
checkpoints. Legacy reconstruction obtains the expected identity from the
current validated dataset and records that no historical cursor was available.

### 5. Runtime and branch identity

Development uses branch `fix/phase1-resumable-loader`, created directly from
the annotated runtime tag `phase1-runtime-2c09840` and commit
`2c098404a3cce0c86f0b863dcd8d3aeb18a55d94`.

The branch is pushed independently. It is not merged into `main` or
`refactor/pi05-libero-bsp-slim` during phase one. After image saving finishes,
the server may check out only the fully verified feature commit. The run
manifest records the pre-interruption runtime SHA, the resume SHA, restored
checkpoint step, cursor source (`legacy-derived` or `checkpoint`), and all
validated loader fields.

## Interfaces

- The loader protocol gains a read-only resume identity and an absolute
  consumed-batch offset used during construction.
- Checkpoint utilities gain cursor load/save helpers scoped to the existing
  assets directory.
- The training entry point computes the accumulation plan before constructing
  the loader, obtains the resume cursor before reading a batch, and passes the
  completed step when saving loader metadata.
- The existing user-facing command remains `scripts/train.py ... --resume`.
  No new required CLI option is introduced.

## Failure handling

Resume aborts before GPU training when:

- the latest checkpoint is incomplete or has an Orbax temporary directory;
- stored step and Orbax step differ;
- consumed batches do not equal `step * accumulation_steps`;
- dataset, sidecar, seed, shuffle, process, or batch topology differs;
- the sampler protocol is unknown;
- the restored step exceeds the requested 10,000 steps;
- exact legacy index parity is not available for the server environment.

Errors report the stored and requested values without modifying the checkpoint.
`--overwrite` remains incompatible with `--resume`.

## Test strategy

### Dependency-free tests

- cursor schema, validation, and JSON round-trip;
- legacy consumed-batch derivation, including step zero;
- mismatch rejection for seed, dataset identity, process count, batch sizes,
  accumulation steps, and sampler protocol;
- checkpoint path selection and missing-cursor behavior;
- no off-by-one between completed optimizer step and next loader batch.

### PyTorch/server tests

- replayable sampler indices exactly equal the existing seeded shuffled loader
  for multiple dataset sizes, batch sizes, epoch boundaries, and offsets;
- uninterrupted and interrupted-plus-resumed runs consume identical index
  sequences;
- `num_workers=0` and the phase-one worker configuration preserve index order;
- a legacy checkpoint resumes at the expected next micro-batch;
- a newly written checkpoint contains a valid cursor and resumes from it;
- malformed or mismatched cursor data fails before the first training batch;
- a small-model gradient test produces equal final state for uninterrupted and
  resumed execution.

### Phase-one server gate

- image creation is complete and the instance reports `Running`;
- `/root`, repository, environments, data mount, sidecar, norm stats, and
  checkpoint directory survived the restart;
- server repository is still at the frozen runtime SHA before deployment;
- the latest complete BSP checkpoint is identified without assuming its step;
- feature commit tests pass in the OpenPI Python 3.11 environment;
- dry resume initializes and reports the expected next batch/step before the
  formal process is launched;
- only one GPU training process runs, with the original short10k protocol.

## Acceptance criteria

- Existing legacy checkpoints remain readable.
- The pinned server environment proves exact old-loader versus replayable-
  sampler index parity.
- A resume at step `N` begins optimization at `N + 1` using the exact next
  loader batch.
- Every new permanent checkpoint contains validated cursor metadata.
- A second interruption resumes from new cursor metadata without manual batch
  calculations.
- Baseline and BSP artifacts remain distinguishable, and the BSP report records
  the runtime SHA transition.
- No changes are made to `main`, the slimming branch, completed baseline
  checkpoints, or the BSP checkpoint being restored.
