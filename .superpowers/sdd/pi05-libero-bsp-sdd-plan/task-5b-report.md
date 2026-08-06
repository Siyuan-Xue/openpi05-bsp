# Task 5B report: phase-one evaluation identity and comparison

## Scope completed

- Bumped the LIBERO evaluation manifest to schema 2 and added required
  `config_name`, integer `checkpoint_step`, and BSP cache manifest fingerprint
  identities. Both cache fields are null for baseline and lowercase SHA256
  values for BSP; `bsp_cache_hash` is explicitly the actual sidecar NPZ hash.
- Added canonical optional task filtering (`0..9`, unique, sorted) and recorded
  it in the manifest, so the same evaluator supports a real
  `libero_spatial`/task-0/one-trial EGL smoke run and the full 40-task protocol.
  The official baseline horizon-10 calibration protocol remains available.
- Added the dependency-free `openpi_client.libero_report` validator and fixed
  comparison engine. It classifies exactly six runs only by schema-2 manifest,
  validates 12,000 paired rollouts and Task-6 cache/norm diagnostics, computes
  task/suite/four-suite macro metrics, and runs the fixed seed-42, 10,000-draw
  task-stratified paired bootstrap at 10k/20k/30k.
- Added an argparse CLI that writes exactly six outputs: task and suite CSVs,
  the three-point learning-curve CSV and SVG, strict comparison JSON, and a
  Markdown report. No checkpoint-selection rule is implemented.

## Test-first evidence

The first client-manifest test run failed in the intended three places:
missing task-filter resolver and missing schema-2 manifest fields. The report
suite then failed because `openpi_client.libero_report` did not exist, and the
CLI suite failed because `scripts.compare_libero_phase1` did not exist. A
subsequent reconstruction-diagnostics RED check proved negative or
misordered mean/p95/max values were not rejected before that validation was
added.

Final dependency-free verification:

```text
PYTHONPATH=packages/openpi-client/src:. python3 -m unittest \
  openpi_client.inference_test \
  openpi_client.libero_eval_test \
  openpi_client.libero_report_test \
  scripts.compare_libero_phase1_test

Ran 30 tests in 6.769s
OK
```

Coverage includes a complete synthetic 12,000-rollout comparison, schema-2
six-run classification, missing/duplicate/extra milestones, official h10 and
wrong-config rejection, strict NaN/truncation handling, artifact/infrastructure
and summary inconsistency rejection, paired identity failures, diagnostic hash
gates, hierarchical macro arithmetic, constant and non-constant deterministic
bootstrap cases, and the exact six-artifact output set.

Also passed:

- `py_compile` for every Task 5B production and test file;
- Python 3.7 AST parsing for both client modules and the comparison CLI;
- Python 3.8 AST parsing for the LIBERO evaluator;
- targeted `git diff --check`.

## Required server gates

- Run `scripts/libero_eval_test.py` inside the isolated LIBERO simulator
  environment, because local NumPy/imageio/LIBERO/pytest dependencies were not
  installed.
- Run one real `libero_spatial`, task-0, one-trial EGL evaluation and inspect
  its schema-2 manifest before the official horizon-10 calibration.
- After all six 2,000-rollout runs and Task-6 diagnostics exist, run the
  comparison CLI against the real artifacts and archive its input hashes and
  six outputs.

No environment was installed or synchronized, and no dataset, checkpoint,
training, policy server, simulator, or container was run locally.

## Independent-review hardening

The independent Task 5B review identified identity and artifact fields that
were present but not yet fully bound to one another. The follow-up now also:

- requires six globally unique checkpoint strings after removing trailing
  slashes and binds each terminal path component to its integer optimizer step;
- validates all ten preparation verification flags, both SciPy-version gates,
  code/cache/rebuilt-content identities, and the complete normalization-state,
  action-hash, asset-directory, and tolerance contract;
- accepts either ordering of reconstruction mean and p95 while requiring both
  to be non-negative and at most the strict maximum below `0.002`;
- reconstructs canonical episode IDs from a lowercase SHA256 simulator-state
  fingerprint and audits step/replan/retry/status/error/timing consistency;
- requires the real `v2.0` dataset identity, lowercase 40/64-character Git SHA,
  Docker `sha256:` digest, and positive finite shared network deadlines.

The expanded dependency-free suite contains 36 tests, including the complete
12,000-rollout fixture and adversarial episode/manifest/diagnostics cases.
