# Deploy the selected EgoVerse ResNet policy on Piper

All deployment-owned files and logs stay below `/mnt/wd10/kerun`. Do not run
the generic `lerobot-record` command for first bring-up: it enables the arm as
part of `connect()` and does not provide the guarded modes below.

## Validated model contract

- checkpoint: `/mnt/wd10/kerun/checkpoints/piper_finetune_geodesic_2726084_step8000.ckpt`
- follower calibration: `/mnt/wd10/kerun/calibration/my_piper_follower.json`
- SHA-256: `c4f57023c804b4a2543fdf539fc14b6c51266b0a670a412d03517f6751b9233f`
- visual encoders: ResNet18 front + ResNet18 wrist
- action frame: absolute Piper base frame (`cartesian_base`); no camera
  extrinsics are applied at deployment
- input cameras: RGB 640x480 at 30 Hz, D435IF front and D405 wrist
- model output: 100x10, representing a 30-frame raw action horizon
- deployment: 10 flow-matching steps, qf5 raw control frames, binary gripper
  threshold 0.5

The qf5 model rows are `[0, 3, 7, 10, 13]`, not the first five rows. The
policy computes this mapping with the same formula as the offline evaluator.

## Environment

Use the existing dependency environment read-only, but force both source
imports to the kerun copies:

```bash
source /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/activate
export PYTHONPATH=/mnt/wd10/kerun/egoverse_piper/EgoVerse:/mnt/wd10/kerun/lerobot/src
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/wd10/kerun/lerobot
```

`source` only reads the dependency environment. All edited source, checkpoints,
fixtures, and generated logs are under `/mnt/wd10/kerun`.

## Hardware-free checks

Load and checksum the checkpoint without constructing a robot:

```bash
python examples/piper/deploy_egoverse_hpt.py --mode checkpoint --device cuda
```

Replay the checked-in local fixture without opening CAN or cameras:

```bash
python examples/piper/deploy_egoverse_hpt.py \
  --mode offline \
  --offline-npz /mnt/wd10/kerun/offline_fixtures/teddy_0912_7_ep0_first10.npz \
  --max-steps 10 \
  --query-frequency 5 \
  --device cuda
```

JSONL audit logs are written to `/mnt/wd10/kerun/deploy_logs`.

## Live dry-run (do not run until the CAN mapping is physically verified)

This mode opens the requested CAN channel and cameras but sets
`enable_motion_on_connect=False`. The SDK wrapper does not call
`EmergencyStop(resume)`, `EnablePiper`, `MotionCtrl_2`, `JointCtrl`, or
`GripperCtrl`; any attempted action send raises an exception.

```bash
python examples/piper/deploy_egoverse_hpt.py \
  --mode live-dry-run \
  --can-port VERIFIED_FOLLOWER_CAN_NAME \
  --max-steps 300 \
  --device cuda
```

Do not guess the CAN name. The observed adapter serial-to-name mapping conflicts
with the repository's old udev example and must be physically confirmed first.

## Execute gate

`execute` is intentionally unavailable without all of the following:

1. explicit `--can-port`;
2. explicit base-frame `--workspace-min=x,y,z` and
   `--workspace-max=x,y,z`;
3. explicit `--allow-motion`;
4. successful offline and live-dry-run review;
5. a person next to the arm with physical emergency stop access.

Initial execution uses `speed_percent=5` and `max_relative_target=2`. The script
performs and logs one discarded warm-up inference before the timed loop and
sends no command during that warm-up.

The deployment robot is configured not to cut torque automatically when the
process closes CAN, because an automatic `DisablePiper()` can make the arm
drop. Follow the approved physical shutdown procedure while supporting the arm.
