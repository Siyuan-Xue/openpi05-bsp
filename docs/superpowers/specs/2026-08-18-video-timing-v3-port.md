# LIBERO video-timing v3 semantic port manifest

## Scope

This manifest records the semantic port of the 22 commits from
`feat/libero-video-timing-v3`, beginning at
`7e1ed7aeaf9ecc1b7c80d701d8202a76400feebf` and ending at
`b12650c62b3df8eec511a0fb1c14aede69a7a050`, onto the slim refactor tree
based at `8420b70d267fbbf2295ea6d28aac07a96ee52e56`.

The port preserves the feature contracts without merging `main` or restoring
the Docker, Compose, and host-test files deleted by the slim refactor. Shared
JSON, CSV, and atomic-write behavior remains owned by
`openpi_client.libero_artifacts`.

## Source-to-destination ledger

| # | Source commit | Status | Slim destination / contract | Reason |
|---:|---|---|---|---|
| 1 | `7e1ed7aeaf9ecc1b7c80d701d8202a76400feebf` Add LIBERO video timing contracts | implemented | `openpi_client/libero_video_timing.py`; `libero_video_timing_test.py` | Retain the dependency-free 20 Hz / 40 fps timing vocabulary, four-clock field naming, integer quantization, and video-audit arithmetic. |
| 2 | `bba54fc1259ca657af08822a204f9d627342ba6f` Harden LIBERO video timing events | implemented | `libero_video_timing.py`; `libero_video_timing_test.py` | Retain strict non-negative integer validation, chronology, non-overlap, schedule/reason matching, and request-latency versus control-stall separation. |
| 3 | `5325746842b1cbdb43ddfd8919411c8452aa5781` Integrate LIBERO video timing artifacts | implemented | `examples/libero/main.py`; `openpi_client/libero_eval.py`; retained evaluator/video tests | Port request and stall measurement, wait-aware video construction/readback, audit persistence, and explicit 20 Hz environment identity into the slim evaluator. |
| 4 | `bb65fb946129a5adbaa160f1a122651b071c509b` Fix LIBERO wait-aware video audits | implemented | `main.py`; `libero_eval.py`; timing/evaluator video tests | Preserve enabled/disabled wait overlays, schedule-specific stall labels, measured-versus-included stall accounting, and the rule that async request latency alone never creates stall frames. |
| 5 | `d1a139d34497a99592104498fddd41828243ea6e` Define LIBERO evaluation schema v3 | implemented | `libero_eval.py`; `libero_report.py`; `main.py`; retained eval/report/script tests; slim runbook | Add schema-v3 timing/video/runtime identities while preserving schema-v2 parsing and the slim README/runbook shape. Assertions formerly placed in deleted host tests move to retained tests. |
| 6 | `8197652b812a2bc0e2d4b773aa52024f85c02520` Harden schema v3 migration gates | implemented | `libero_report.py`; `libero_report_test.py`; retained host contract | Require schema-specific field sets and reject partial/mixed v2/v3 manifests without recreating the removed server test. |
| 7 | `da96c565cbda77a533eb45346717094de29454cb` Require exact manifest JSON containers | implemented | `libero_report.py`; `libero_report_test.py` | Require exact JSON object/array container types (not generic mappings/sequences) at manifest boundaries. |
| 8 | `4045d4d99bc6cf45b182b26c25b2d3a114473903` Restore manifest test scope | superseded by slim equivalent | retained `libero_report_test.py` manifest fixtures and schema-v2/v3 cases | The source commit only repaired test placement/scope; the port expresses the same coverage in the retained pytest module. |
| 9 | `1b32660637610999ea6781f5bae27ea3ec3089e7` Fix LIBERO schema migration boundaries | implemented | `libero_eval_test.py`; `libero_report.py`; `libero_report_test.py` | Keep v2 defaults legal, require v3-only identities for v3, and reject version/field cross-contamination. |
| 10 | `b6d43ad852b5434b3eca0b7b4500364a04011da1` Fix LIBERO failure video artifacts | implemented | `main.py`; `libero_eval.py`; retained evaluator/video tests | Selected policy-failure episodes keep their frames/timing and produce audited video artifacts without changing metric classification. |
| 11 | `9b6dfbc63d97bafc731688e5017b4cee960aa233` Harden LIBERO runtime identity gates | superseded by slim equivalent | `main.py` automatic identity resolution; `tests/contracts/libero_host_contract_test.py`; slim runbook | Runtime/Git identity remains fail-closed in retained code and tests; edits to deleted `Dockerfile`, `compose.yml`, and server test are intentionally not restored. |
| 12 | `16e963eedc1e3dfe4271c7781bbb9c1f59c9f623` Preserve legacy videos after policy failure | implemented | `main.py`; retained `scripts/libero_eval_test.py` | Preserve the already-captured replay frames when policy inference fails so selected legacy/failure videos remain encodable. |
| 13 | `7155efd7b69764ef6a06589eb03b03d370b09e5e` Fail closed on LIBERO runtime identity | superseded by slim equivalent | `main.py`; retained host contract and runbook | The slim evaluator rejects missing/unresolvable runtime identity. Source assertions in the deleted server test are represented by retained host/evaluator tests. |
| 14 | `4587c4fb84f31af88c613910b21c8cfe5dea1bd0` Encode zero-step LIBERO failure videos | implemented | `main.py`; `libero_eval.py`; retained evaluator/video tests | Capture a source frame before the first inference so a replan-zero policy failure still yields a one-frame, auditable video. |
| 15 | `178905811c049d1c099dccf856fdcd5d25ca35e1` Strengthen LIBERO failure artifact tests | implemented | retained `scripts/libero_eval_test.py` | Port the observable zero-step and later policy-failure video cases into the retained pytest test surface. |
| 16 | `473cb59a100fa9ae86abddb896b1dccaaeb9d652` Align manifest validation exception contract | implemented | `libero_eval_test.py` | Keep manifest construction validation consistently exposed as `ValueError` at the public contract. |
| 17 | `69d7b26c7e2d6b872842e0516622d139b5e25c2f` Format LIBERO video timing modules | superseded by slim equivalent | final ported timing/evaluator modules and tests | This is formatting-only provenance; the semantic port uses the repository's current formatting without a separate behavior change. |
| 18 | `7a1abcb42b26724db949e8cf34ad712bc89c6690` Document Python 3.8 zip compatibility | implemented | `main.py` wait-frame pairing comment and non-strict `zip` usage | Preserve Python 3.8-compatible iteration; do not use `zip(..., strict=True)` in the client/evaluator path. |
| 19 | `e3fb0b1199e716390ad49764b9b300fe5de345fe` Require automatic evaluator Git identity | implemented | `main.py`; retained `scripts/libero_eval_test.py` | Resolve evaluator Git SHA automatically and fail closed when the checkout identity cannot be established. |
| 20 | `2fd655cdd193226dd753572e96dbaa2d3dc56ecd` Fail closed on missing stall source frames | implemented | `main.py`; retained `scripts/libero_eval_test.py` | Reject video construction when a measured stall cannot be anchored to the exact captured source frame. |
| 21 | `e4a7ea86874397296088132380b53c06e8eb0e49` Audit LIBERO video stall overlays explicitly | implemented | `libero_eval.py`; retained `libero_eval_video_test.py` | Persist explicit overlay request/render counts and fail the audit if requested stall overlays were not rendered exactly. |
| 22 | `b12650c62b3df8eec511a0fb1c14aede69a7a050` Harden schema-v3 identity and video audits | implemented | `libero_eval.py`; `libero_report.py`; retained report/video/host tests; slim runbook | Fail closed on schema-v3 evaluator Git/runtime identity, missing/extra audit records, malformed timing events, missing stall-source frames, and incomplete overlay evidence. |

## Intentionally omitted source surface

The following slim deletions remain deleted:

- `examples/libero/Dockerfile`
- `examples/libero/compose.yml`
- `scripts/pi05_libero_bsp_phase1_server_test.py`
- `scripts/server_runtime_contract_test.py`

No unrelated `main` dependencies, generated artifacts, caches, or environment
files are part of this port. Any still-relevant host assertions are expressed
through retained evaluator/report/contract tests.
