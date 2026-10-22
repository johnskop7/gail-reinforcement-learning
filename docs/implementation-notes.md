# Implementation and preservation notes

## Main study versus exploratory material

The February 2025 report focuses on CartPole, Pendulum, BipedalWalker, and LunarLander, with TRPO/PPO/GAE-labeled expert variants. The package includes the associated checkpoints and reward logs. Additional environment configurations and Q-learning code are preserved, but do not establish that every environment/algorithm combination was completed.

The main GAIL source is taken from the supplied `gail/gail` folder. The earlier `gail-pytorch-main` tree remains under `reference/gail-baseline/`. Its original README, figures, source, and license are retained; its original absolute/root asset links and Windows paths describe that historical copy, not the active repository workflow. The isolated `validate_gym` script is retained under `reference/standalone-validation/` and is not an independent runnable package.

## Packaging edits

- Flattened the nested main GAIL folder into `imitation_learning/`.
- Preserved `expert_policies/`, including its model/config/checkpoint hierarchy.
- Replaced personal Windows paths in active scripts with paths relative to `__file__`.
- Resolved expert training/evaluation checkpoint and config paths relative to their scripts.
- Added `--expert_name` and `--run_name` to GAIL training to match the supplied expert-variant folder names and separate new outputs.
- Rejected pre-existing GAIL output directories rather than overwriting saved runs.
- Preserved the original versions of every edited script under `reference/original-scripts/`.
- Added documentation, root dependency inventory, artifact inventory, provenance, and GitHub settings.
- Omitted compiled Python cache files only; no learning algorithms, configurations, model weights, or reward logs were removed or rewritten.

See `provenance.json` for the source-to-destination mapping and hashes, including changed-script hashes. The original report is copied byte-for-byte to `docs/report.pdf`.

## Reproduction limits visible in the source

- Legacy Gym environment IDs and the older reset/step/render contract require a compatible runtime.
- Neither the supplied dependency lists nor this packaging step establish a tested environment lockfile.
- The main networks use three hidden layers of 50 units; introductory architectural descriptions in the report should be distinguished from the concrete saved implementation.
- The `gae` filename labels a particular optimizer using Generalized Advantage Estimation. GAE itself is an advantage-estimation method, also used within other algorithms.
- The GAIL FrozenLake/Q-table path is incomplete and is not represented as a supported example.
- Some checkpoint folders contain only configuration. The file inventory distinguishes these from folders with policies.
- The `.npy` extension on `q_table.npy` does not guarantee NumPy file format: its leading bytes identify a ZIP container, as with the Torch-saved Q-table artifacts. Do not infer a loader from the filename alone.
- Some validation scripts use non-strict parameter loading, and action selection differs between scripts (sampling versus argmax/mean).
- The plotting code's standard-deviation shading is computed within each reward trace. It is not a multi-seed confidence interval.

No simulator runs or training were performed. Validation covers source syntax, configuration parsing, active path rewrites, referenced-file existence, artifact hashes, and ZIP integrity.
