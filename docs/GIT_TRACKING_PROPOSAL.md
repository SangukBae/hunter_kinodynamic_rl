# Git-tracking proposal for `hunter_kinodynamic_rl/` (historical)

> **Historical artifact, not current Git instructions.** The staged state
> described below was true for round 5. At the 2026-09-02 documentation audit,
> `git diff --cached --shortstat` is empty, `HEAD` is
> `4d2cbf2 Complete hierarchical navigation phases 2 and 3`, and this package
> again has modified and untracked files. Review the live `git status` and
> diffs afresh before staging or committing; do not assume the old 197-file
> staging set still exists.

> **2026-09-02 release update:** the package source/config/tests/docs were
> reviewed, generated `runtime/` artifacts remained ignored, and the verified
> corrected baseline was committed and annotated as
> `hunter-kinodynamic-rl-r0-20260902`. The proposal below remains historical;
> it is not a request to restage or recreate that release.

**Historical round-5 status: applied at the staging level only.** This
proposal's exact plan (the `.gitignore` line below, then `git add
ros2_ws/src/hunter_kinodynamic_rl`) has now been executed so that
`git diff --cached` / `git status` can actually show reviewable content
for this package instead of a single opaque `?? ...` line. **No commit
was made** — `git log` is unchanged (still `968c9f9 ...`), and nothing
was pushed. The pre-existing, unrelated working-tree modifications to
`.gitignore` (the `ros2_ws/runtime/` line already present before this
round) and `ros2_ws/src/drl_agent_interfaces/package.xml` were left
byte-identical and untouched; `.gitignore` itself was also deliberately
left UNSTAGED (only the one new line was added to its working-tree
content) so this session's staging action doesn't fold that unrelated,
already-modified file into anything.

Verified after staging:
- `git status --short | grep -c hunter_kinodynamic_rl/runtime` → `0`
  (the new `.gitignore` line correctly excludes it)
- `git status --short | grep -c __pycache__` → `0`
- 197 files, 30,491 insertions staged (`git diff --cached --shortstat --
  ros2_ws/src/hunter_kinodynamic_rl`)
- No files outside `ros2_ws/src/hunter_kinodynamic_rl/` were staged

Everything below this line is the original (round-1) proposal text,
preserved as-is since the plan it describes is what was actually
executed.

## What should be committed

Everything under `ros2_ws/src/hunter_kinodynamic_rl/` is source EXCEPT the
generated/ephemeral directories listed below. Concretely:

```
hunter_kinodynamic_rl/
├── CMakeLists.txt              # commit
├── package.xml                  # commit
├── conftest.py                  # commit
├── pytest.ini                   # commit
├── README.md                    # commit
├── resource/                    # commit (ament resource marker)
├── hunter_kinodynamic_rl/       # commit (the importable Python package)
│   └── **/*.py                  # commit
│   └── **/__pycache__/          # NEVER commit (see below)
├── config/                      # commit (profiles/benchmarks/robot/etc YAML)
├── launch/                      # commit (*.launch.py)
├── tests/                       # commit (*.py)
│   └── __pycache__/             # NEVER commit
├── docs/                        # commit (*.md)
└── runtime/                     # NEVER commit (see below) -- checkpoints,
                                  # training run output, verification logs
```

This mirrors how the sibling `drl_agent` package is already tracked in
this same repo (source + config + tests + docs committed, its own
`runtime/` excluded via the existing root `.gitignore` line
`ros2_ws/src/drl_agent/runtime/`) -- `hunter_kinodynamic_rl` should follow
the identical convention, not invent a new one.

## What should NEVER be committed

| Path pattern | Why |
|---|---|
| `hunter_kinodynamic_rl/runtime/` (the WHOLE directory) | Training run output (checkpoints -- `.pt`/`.npz` files, often tens of MB each -- JSONL/CSV logs, TensorBoard event files) and this session's own `runtime/verification/<timestamp>/` artifacts. All regenerable by re-running training/tests; none of it is source. Committing even ONE checkpoint would make every future clone/fetch download that binary blob forever (git does not shrink history on deletion without a rewrite). |
| `**/__pycache__/` | Already covered by the EXISTING root-level `.gitignore` (`__pycache__/`, `*.pyc`, no leading `/` so it matches at any depth) -- no new rule needed. |
| `.pytest_cache/` | Already covered by the existing root `.gitignore`. |
| `system_id_results/` (if ever produced under this package's own working directory rather than under `runtime/`) | Real-trial CSV/JSON output from `system_id_node.py` -- regenerable, run-specific, same class of artifact as `runtime/`. |
| Any `*.pt`, `*.npz`, `*.tfevents.*` anywhere under this package | Belt-and-suspenders pattern in case a future script writes one outside `runtime/` by mistake -- these are always checkpoint/replay/TensorBoard binary artifacts in this codebase, never source. |

## Proposed `.gitignore` change

The repo already has a ROOT-level `.gitignore` (not a per-package one) that
lists each ROS package's own `runtime/`-equivalent directory individually
(`ros2_ws/src/drl_agent/runtime/`, `ros2_ws/src/drl_agent/runs/`,
`ros2_ws/runtime/`, ...). The proposal is to follow that SAME convention
-- add one line, in the same place those other `runtime/` lines already
live:

```diff
 ros2_ws/src/drl_agent/runtime/
 ros2_ws/src/drl_agent/runs/
 ros2_ws/runtime/
+ros2_ws/src/hunter_kinodynamic_rl/runtime/
 /runs/
```

No package-local `.gitignore` file is proposed inside
`hunter_kinodynamic_rl/` itself -- every other package in this repo is
excluded via the root `.gitignore`, and introducing a second convention
for just this one package would be inconsistent with the rest of the
tree for no real benefit.

## Suggested commit sequence (when the user is ready)

Not executed by this session -- offered only as the concrete sequence a
maintainer could follow:

```bash
# 1. Add the proposed .gitignore line first, so step 2 never even
#    considers runtime/ for staging.
#    (edit .gitignore as shown in the diff above)

# 2. Review exactly what would be staged BEFORE staging it -- especially
#    worth checking nothing under runtime/ slipped through and nothing
#    unexpectedly large is about to be committed.
git add -n ros2_ws/src/hunter_kinodynamic_rl
git status --porcelain ros2_ws/src/hunter_kinodynamic_rl | grep -v '^?? ros2_ws/src/hunter_kinodynamic_rl/runtime/'

# 3. Stage and commit.
git add ros2_ws/src/hunter_kinodynamic_rl .gitignore
git commit -m "Add hunter_kinodynamic_rl package"
```

## Explicitly out of scope for this proposal

- Whether/when to actually start tracking this package -- a decision only
  the user should make, not inferred from this task.
- Splitting the initial commit into smaller pieces (e.g. one commit per
  subsystem) -- a single, whole-package initial commit is the simplest
  starting point; a maintainer preferring a different history shape can
  restructure this proposal's single `git add` into several before
  committing.
- Any migration of the EXISTING `runtime/p5_live_check/` (a flat, legacy-
  layout checkpoint from an earlier session, predating the item-3
  checkpoint-generation rewrite) or `runtime/verification/` directories --
  both stay exactly where they are, on disk, gitignored, untouched by this
  proposal.
