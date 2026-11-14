# M5 demonstration dataset

One HDF5 file of successful episodes. The retargeted episodes are **部分重定向
+ 合成放置** (partial retarget plus a synthetic place): the approach from
home, the place slide, the descent, and the retreat are synthetic; the
pre-grasp positions are the human can-relative offset; the carry is a
synthetic baseline plus the human residual. Orientation is rewritten to a
top-down yaw, and the gripper timing is rewritten to the M4 waits, ramp, and
dwell. Paths below are HDF5 groups. `T` is the number of 20 Hz control ticks,
ending on the tick where the M1 success test first stays true.

## Root attributes

| attribute | meaning |
|---|---|
| `control_hz` | 20 |
| `camera_names` | JSON list, `agentview` and `wrist_cam_left` |
| `image_height`, `image_width` | 96 |
| `action_convention` | base frame, xyz metres, quaternion wxyz, absolute TCP pose, gripper opening metres. Action `t` is the pose commanded for tick `t+1`. `next_actions[t]` is the preview that was actually passed to `env.step` (the last tick included). |
| `reference_mode` | `preview` (`env.step(..., next_action=...)`) |
| `method` | `部分重定向 + 合成放置` |
| `git_commit` | `git rev-parse HEAD` |
| `git_dirty` | `true` when `git status --porcelain` was non-empty |
| `file_sha256` | JSON map of path to sha256 for `src/aloha_pp/data/`, `planning/noise.py`, `sim/env.py`, `scripts/m5_generate.py`, `scripts/m4_run.py`, and the M1 / M3 / M4 / M5 configs |
| `config_json` | snapshot of `configs/m5_retarget.yaml`, `configs/m5_noise.yaml`, and M4 `drop_z` |
| `n_demos` | number of `demo_*` groups |

## `data/demo_XXXX`

Observations are the state **before** `actions[t]`.

| dataset | shape | dtype |
|---|---|---|
| `obs/agentview` | `T × 96 × 96 × 3` | uint8 |
| `obs/wrist_cam_left` | `T × 96 × 96 × 3` | uint8 |
| `obs/tcp_pose` | `T × 7` | float32, xyz + wxyz in the base frame |
| `obs/gripper_opening` | `T × 1` | float32, metres |
| `obs/q` | `T × 6` | float32, arm joints |
| `obs/qvel` | `T × 6` | float32 |
| `privileged/can_pose` | `T × 7` | float32, collision-cylinder pose in the base frame |
| `actions` | `T × 8` | float32, xyz + wxyz + gripper. These are the commands that were executed. |
| `next_actions` | `T × 8` | float32, the preview passed on that tick |
| `phase` | `T` | int8, segment of `actions[t]`. See below. |
| `qpos0`, `qvel0`, `ctrl0` | model width | float64, state after reset, before the first action |
| `qacc_warmstart0`, `qacc0` | `nv` | float64. Restoring the warm start makes a replay bit-exact. |
| `time0` | attribute | float, simulator time after the settle |

Attributes on the group: `source` (`retarget` or `planner_noise`), `source_id`
(robomimic demo index, or the planner seed), `slowdown` (time-stretch factor;
1 for an un-noised planner demo), `success` (always true in this file),
`success_step`, `phase_scheme` (`retarget` or `planner`), `human_fraction`.

### `phase` when `phase_scheme` is `retarget`

| value | name | what the tick is |
|---|---|---|
| 0 | `synth_approach` | quintic from the home TCP; the first pose sample is home |
| 1 | `human_pregrasp` | human position relative to the can, on this project's can |
| 2 | `grip_wait` | hold while the preview catches up and the grasp settles |
| 3 | `grip_ramp` | linear gripper ramp |
| 4 | `grip_dwell` | hold after the ramp |
| 5 | `carry` | synthetic baseline plus the human residual |
| 6 | `place_slide` | synthetic slide over the cell |
| 7 | `place_down` | synthetic descent |
| 8 | `retreat` | synthetic retreat |

`human_fraction` is the share of stored ticks whose phase is
`human_pregrasp` or `carry`. Those ticks contain a human position or a human
carry residual. The carry is not purely human: it also has the synthetic
baseline. Gripper waits, ramps, and dwells repeat a pose, so they are not
counted. Each motion segment is slowed by its own preview peak, with a
quintic blend of the slowdown at the boundary. The gripper blocks are
inserted after that scaling and keep the M4 durations (one lag tick plus
`settle_s`, a 12-tick ramp, 0.40 s closed, 0.20 s open). The fraction is
measured on the stored sequence, after the segment scaling and after the
success cut. Orientation and gripper timing are rewritten, as above.

### `phase` when `phase_scheme` is `planner`

The value is the M4 schedule state (`approach` = 0 through `wait_success` = 14
in `aloha_pp.planning.schedule.STATE_NAMES`). Those episodes are fully
synthetic. `human_fraction` is not meaningful for them.

Read an episode with `aloha_pp.data.dataset.load_episode`. Check a file with
`scripts/m5_validate.py`.
