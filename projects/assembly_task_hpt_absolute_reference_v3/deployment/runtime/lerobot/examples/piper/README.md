# Piper: Recording Demonstrations

Step-by-step guide to recording teleoperated demonstrations on a Piper
leader/follower pair with lerobot. See [`command`](../../command) at the repo
root for copy-pasteable versions of every command below.

## Prerequisites

- Two Piper arms wired as a leader/follower pair, each on its own USB-CAN
  adapter.
- The `lerobot` conda env, with this repo's Piper robot/teleoperator
  registered (`src/lerobot/robots/piper`, `src/lerobot/teleoperators/piper_leader`).
- Camera(s) plugged in (RealSense in the examples below).

## Step 1 — Bring up the CAN interfaces

If [`udev/99-piper-can.rules`](../../udev/99-piper-can.rules) and the
`can-follower.service` / `can-leader.service` units are installed, each
adapter is auto-activated (fixed name + bitrate) as soon as it's plugged in —
skip to Step 2 and just confirm both are up:

```bash
ip -details link show type can
```

Otherwise activate manually, once per adapter:

```bash
bash /mnt/wd10/tongxi/Piper_sdk_ui/can_activate.sh can_leader   1000000 "1-3.1.3:1.0"
bash /mnt/wd10/tongxi/Piper_sdk_ui/can_activate.sh can_follower 1000000 "1-3.1.4:1.0"
ip -details link show type can   # confirm both interfaces are UP
```

The third argument is each adapter's physical USB port path — find yours
with `dmesg` if you re-plug cables into different ports (or use the udev
approach above, which keys off each adapter's fixed USB serial instead so
port order never matters).

## Step 2 — Calibrate joint limits (once per arm, or whenever unsure)
(Already done, don't need to do it now. But if the robot is resetted, this is needed to be done)
Move the arm by hand through its full range while the script logs real
min/max joint angles — torque is disabled throughout, so this is safe. Do
**not** trust datasheet limits or the SDK's live query blindly: both arms
have been found to report generic, uncalibrated defaults on this hardware.

```bash
python examples/piper/find_real_joint_limits.py --port=can_follower
python examples/piper/find_real_joint_limits.py --port=can_leader --source=ctrl
```

The leader needs `--source=ctrl` — see the script's docstring for why. Once
you have real numbers, trim a few degrees in from each hard stop as margin,
and keep the two arms' results handy for `--robot.joint_limits_deg` /
`--teleop.joint_limits_deg` below.

Optional sanity checks along the way:
- [`read_gripper_stroke.py`](read_gripper_stroke.py) — measure each arm's actual
  gripper full-open stroke in mm (don't assume leader == follower == datasheet).
- [`verify_follower.py`](verify_follower.py) / [`verify_leader.py`](verify_leader.py) —
  basic connectivity/behavior checks per arm.

## Step 3 — Test teleoperation

Drive the follower live with the leader before recording anything, to
confirm motion direction, speed, and limits all look right:

```bash
lerobot-teleoperate \
  --robot.type=piper --robot.port=can_follower --robot.speed_percent=15 \
  --robot.gripper_stroke_mm=102.8 \
  --robot.joint_limits_deg='[[-151.0,151.0],[0.9,186.0],[-170.1,-1.0],[-124.2,124.9],[-85.0,87.7],[-168.4,168.4]]' \
  --robot.id=my_piper_follower \
  --teleop.type=piper_leader --teleop.port=can_leader \
  --teleop.gripper_stroke_mm=104.9 \
  --teleop.joint_limits_deg='[[-151.0,151.0],[3.0,177.0],[-167.0,-3.0],[-124.0,124.0],[-86.5,86.5],[-167.0,167.0]]'
```

Substitute the `joint_limits_deg` / `gripper_stroke_mm` values you measured
in Step 2 for your own arms. Keep `speed_percent` low (10–20) until you're
confident nothing is inverted or mis-limited.

If you ever need to freely move the follower by hand (e.g. to reset its
pose), release its torque:

```bash
conda activate lerobot
python -c "
from piper_sdk import C_PiperInterface_V2
import time
piper = C_PiperInterface_V2('can_follower')
piper.ConnectPort()
time.sleep(0.1)
while piper.DisablePiper():
    time.sleep(0.01)
print('Torque released.')
"
```

## Step 4 — Record a demonstration dataset

With teleoperation confirmed, record real episodes. The leader drives the
follower while both cameras capture video; each episode is saved into a
dataset and (optionally) pushed to the Hub.

```bash
lerobot-record \
  --robot.type=piper \
  --robot.port=can_follower \
  --robot.id=my_piper_follower \
  --robot.speed_percent=100 \
  --robot.cameras='{
    "435if": {"type": "intelrealsense", "serial_number_or_name": "327122075831", "color_mode": "RGB", "use_depth": true, "rotation": 0, "warmup_s": 5, "width": 640, "height": 480, "fps": 30},
    "405":   {"type": "intelrealsense", "serial_number_or_name": "427622273943", "color_mode": "RGB", "use_depth": true, "rotation": 0, "warmup_s": 5, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=piper_leader \
  --teleop.port=can_leader \
  --teleop.id=my_piper_leader \
  --dataset.repo_id=Data_Collect/teddy_0912_1\
  --dataset.single_task="Pick up the teddy bear in the basket" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=true \
  --dataset.private=false \
  --display_data=true
```

Notes:
  --dataset.episode_time_s=60 \: recording period is 60s, you have to finish 
  --dataset.reset_time_s=15 \
 --robot.speed_percent=30: full speed is 100, for safety, it is lower than 100.
- dataset.repo_id=Data_Collect/test_teddy_1\: will save data under "/home/master-ht/.cache/huggingface/lerobot/Data_Collect"
- Camera `serial_number_or_name` values pin specific physical RealSense units
  — needed when more than one is plugged in. Find yours with
  `rs-enumerate-devices` or swap in the generic device name if you only have one.
- `--dataset.episode_time_s` / `--reset_time_s` set the recording window and
  the pause between episodes (to reset the scene) respectively.
- Set `--dataset.push_to_hub=false` to keep the dataset local only; it still
  lands in the local HF cache either way, e.g.
  `~/.cache/huggingface/lerobot/<hf_username>/<dataset_name>`.
- `--display_data=true` opens a live viewer of the cameras/state while
  recording — useful for the first few episodes, drop it once you trust the setup.
   --dataset.single_task: you can redefine your language task. 

## Step 5 — Review what you recorded

Replay a recorded episode on the real follower arm to sanity-check it looks
right physically (no policy/teleop involved — it just plays back the logged
actions):

```bash
lerobot-replay \
  --robot.type=piper \
  --robot.port=can_follower \
  --robot.id=my_piper_follower \
  --robot.speed_percent=30 \
  --dataset.repo_id=<hf_username>/<dataset_name> \
  --dataset.episode=0
```

## Step 6 — Deploy a trained policy

Once you have a dataset you're happy with, train a policy on it (see the
top-level lerobot docs for `lerobot-train`). Deployment reuses the exact same
`lerobot-record` flow as Step 4, except the follower is driven by a
`--policy.*` instead of `--teleop.*` (no leader arm needed), and the run is
saved as a small rollout dataset for review — keep `--dataset.num_episodes`
low (e.g. 1) for a quick check.

### ACT

Run in the `lerobot` conda env. Complete example, exactly as used on this
machine:

```bash
conda run -n lerobot lerobot-record \
  --robot.type=piper \
  --robot.port=can_follower\
  --robot.id=my_piper_follower\
  --robot.speed_percent=15 \
  --robot.max_relative_target=8 \
  --robot.cameras='{
    435if: {"type": "intelrealsense", "serial_number_or_name": "Intel RealSense D435IF", "width": 1280, "height": 720, "fps": 30},
    405:   {"type": "intelrealsense", "serial_number_or_name": "Intel RealSense D405",   "width": 848,  "height": 480, "fps": 30}
  }' \
  --policy.path=/mnt/wd10/tongxi/lerobot/outputs/train/act_teddy_bear_basket_finetune/checkpoints/040000/pretrained_model \
  --dataset.repo_id=yukk1/eval_piper_act_rollout_review \
  --dataset.num_episodes=1 \
  --dataset.single_task="Put the teddy bear into the basket" \
  --display_data=true
```

`--policy.path` points at a `pretrained_model/` checkpoint directory produced
by `lerobot-train`. On this device, ACT checkpoints live under:

```
/mnt/wd10/tongxi/lerobot/outputs/train/<run_name>/checkpoints/<step>/pretrained_model
```

e.g. the `040000` step of the `act_teddy_bear_basket_finetune` run used above.
Swap in your own `<run_name>`/`<step>`, `--dataset.repo_id`, and
`--dataset.single_task` for a different checkpoint or task.

### EgoVerse HPT

This policy lives in `src/lerobot/policies/egoverse_hpt` and needs the
separate **emimic** venv, not the `lerobot` conda env — source it first.
Multiple similarly-named emimic venvs exist on this machine; verify this is
the correct training/deployment one (e.g. check `VIRTUAL_ENV` of a live
process) before trusting it blindly.

The venv, on this device, is at:

```
/mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic
```

Complete example, exactly as used on this machine:

```bash
source /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/activate
lerobot-record \
  --robot.type=piper \
  --robot.port=can_follower \
  --robot.id=my_piper_follower \
  --robot.speed_percent=15 \
  --robot.max_relative_target=8 \
  --robot.cameras='{
    435if: {"type": "intelrealsense", "serial_number_or_name": "Intel RealSense D435IF", "width": 1280, "height": 720, "fps": 30},
    405:   {"type": "intelrealsense", "serial_number_or_name": "Intel RealSense D405",   "width": 848,  "height": 480, "fps": 30}
  }' \
  --policy.type=egoverse_hpt \
  --policy.ckpt_path=/mnt/wd10/tongxi/egoverse_piper/EgoVerse/egomimic/logs/hpt_cotrain_piper_human/wristframe_fullfps_2026-08-22_02-13-35/checkpoints/last.ckpt \
  --policy.action_frame=wristframe \
  --policy.joint_limits_deg='[[-154.6,150.0],[-2.1,191.1],[-173.4,2.0],[-135.4,119.8],[-88.8,89.9],[-170.6,172.3]]' \
  --policy.device=cuda \
  --dataset.repo_id=yukk1/eval_piper_hpt_rollout_review \
  --dataset.num_episodes=1 \
  --dataset.single_task="Put the teddy bear into the basket" \
  --display_data=true
```

Extra `--policy.*` flags vs. ACT:
- `--policy.type=egoverse_hpt` — selects the HPT policy class
- `--policy.ckpt_path=...` — path to the HPT checkpoint (`.ckpt`). On this
  device, HPT checkpoints live under
  `/mnt/wd10/tongxi/egoverse_piper/EgoVerse/egomimic/logs/<run_name>/checkpoints/`
  (e.g. `last.ckpt` from the `hpt_cotrain_piper_human/wristframe_fullfps_2026-08-22_02-13-35` run above).
- `--policy.action_frame=wristframe` — action space the policy was trained in
- `--policy.joint_limits_deg=...` — joint limits used by the policy/robot
- `--policy.device=cuda` — run inference on GPU

Swap in your own `--policy.ckpt_path`, `--dataset.repo_id`, and
`--dataset.single_task` for a different checkpoint or task.

## Deleting unwanted local datasets

Every `lerobot-record` run (demonstrations and rollout reviews alike) lands
in the local HF cache regardless of `--dataset.push_to_hub`:

```
~/.cache/huggingface/lerobot/<hf_username>/<dataset_name>
```

On this device that's `/home/master-ht/.cache/huggingface/lerobot/yukk1/`,
which currently holds the demonstration datasets from Step 4 (e.g.
`pick_and_place_v1_10_2`, `pick_and_place_all`) alongside the rollout-review
datasets from Step 6 (e.g. `eval_piper_act_rollout_review`,
`eval_piper_hpt_solo_camframe`) — for example:

```
/home/master-ht/.cache/huggingface/lerobot/yukk1/pick_and_place_v1_10_2
/home/master-ht/.cache/huggingface/lerobot/yukk1/eval_piper_hpt_solo_camframe_maxrel20_2
```

To discard a whole dataset you don't want to keep (e.g. a botched recording,
or a one-off rollout-review dataset from Step 6), just remove that folder:

```bash
rm -rf ~/.cache/huggingface/lerobot/<hf_username>/<dataset_name>
# e.g. rm -rf /home/master-ht/.cache/huggingface/lerobot/yukk1/eval_piper_hpt_solo_camframe_maxrel20_1
```

If it was already pushed to the Hub and you want it gone there too, delete
the repo from the Hub UI or with `huggingface-cli repo delete
<hf_username>/<dataset_name> --repo-type dataset` — deleting the local folder
does **not** touch the Hub copy.

To drop only specific bad episodes from an otherwise-good dataset instead of
deleting the whole thing, use `delete_episodes()` in
[`src/lerobot/datasets/dataset_tools.py`](../../src/lerobot/datasets/dataset_tools.py)
rather than hand-editing the cache folder — it re-indexes the remaining
episodes/videos/metadata consistently, which a raw `rm` on individual files
would not.
