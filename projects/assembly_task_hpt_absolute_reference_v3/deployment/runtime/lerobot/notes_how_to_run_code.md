# Set up the environment

```bash
conda create -y -n lerobot python=3.10

conda activate lerobot

conda install ffmpeg -c conda-forge

git clone https://github.com/huggingface/lerobot.git

cd lerobot

pip install -e .
```

(use wandb to track the training): 
wandb login

(In this session, you may need to use classis token to git clone the respository)



# Robot Teleoperate and caliboration

```bash
python src/lerobot/scripts/lerobot_teleoperate.py  \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
  --robot.cameras='{
    "top": {
        "type": "intelrealsense",
        "serial_number_or_name": "218622272943",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "right": {
        "type": "intelrealsense",
        "serial_number_or_name": "230322276800",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "low": {
        "type": "intelrealsense",
        "serial_number_or_name": "218622276584",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    }
  }' \
  --teleop.type=widowx \
  --teleop.port=/dev/ttyDXL_leader_left \
  --teleop.id=left_leader \
  --teleop.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx \
  --display_data=true

  
  # (The calibration file will be generated automatically in the first time you try teleoperate (You can find the address in the command)  If you are unsatified with it, just delete the file and re-teleoperate)

#   lerobot-teleoperate \
#   --robot.type=widowxai_follower_robot \
#   --robot.ip_address=192.168.1.4 \
#   --robot.id=follower \
#   --teleop.type=widowxai_leader_teleop \
#   --teleop.ip_address=192.168.1.2 \
#   --teleop.id=leader \
#   --display_data=false

# lerobot-teleoperate \
#   --robot.type=widow_ai_follower \
#   --robot.port=/dev/ttyDXL_follower_left \
#   --robot.id=left_follower \
#   --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
#   --teleop.type=widow_ai_leader \
#   --teleop.port=/dev/ttyDXL_leader_left \
#   --teleop.id=left_leader \
#   --teleop.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx \
#   --display_data=true


# lerobot-record \
#   --robot.type=widow_ai_follower \
#   --robot.port=192.168.1.4 \
#   --robot.id=left_follower \
#   --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
#   --robot.max_relative_target=100.0 \
#   --teleop.type=widow_ai_leader \
#   --teleop.port=192.168.1.2 \
#   --teleop.id=left_leader \
#   --teleop.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx \
#   --display_data=true \
#   --dataset.push_to_hub=false \
#   --dataset.repo_id=lerobot/pick_sponge10 \
#   --dataset.episode_time_s=45 \
#   --dataset.reset_time_s=15 \
#   --dataset.num_episodes=10 \
#   --dataset.single_task="Grab the cube"
```

# caliboration

```bash
# if you donot have any caliboration, /data/sichao/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx/ does not left_leader.json and  /data/sichao/lerobot/.local/huggingface/lerobot/calibration/robots/viperx/ doesnot left_follower.json
# if you want to do the re-caliboration, you need to removed left_leader.json and left_follower.json. 

# The caliboration processs info 
# INFO 2025-12-03 17:22:29 x/widowx.py:119 
# Running calibration of left_leader WidowX
# Move arm to a mid-range 'neutral' pose (all joints roughly centered) and press ENTER...
# Now slowly sweep EACH of these joints through their safe ranges while data is recorded:
# ['shoulder', 'shoulder_shadow', 'elbow', 'elbow_shadow', 'wrist_angle', 'gripper']
# Press ENTER to stop recording...

# -------------------------------------------
# -------------------------------------------
# NAME            |    MIN |    POS |    MAX
# shoulder        |    814 |    822 |   3182
# shoulder_shadow |    762 |    769 |   3126
# elbow           |    646 |   3180 |   3180
# elbow_shadow    |    553 |   3082 |   3085
# wrist_angle     |   1181 |   2397 |   3522
# gripper         |    154 |   2254 |   2255
# INFO 2025-12-03 17:23:05 x/widowx.py:176 Calibration saved to /data/sichao/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx/left_leader.json
# INFO 2025-12-03 17:23:05 x/widowx.py:103 left_leader WidowX connected.
# INFO 2025-12-03 17:23:05 x/viperx.py:142 
# Running calibration of left_follower ViperX
# Move arm to a mid-range 'neutral' pose (all joints roughly centered) and press ENTER...
# Now slowly sweep EACH of these joints through their safe ranges while data is recorded:
# ['shoulder', 'shoulder_shadow', 'elbow', 'elbow_shadow', 'wrist_angle', 'gripper']
# Press ENTER to stop recording...

# -------------------------------------------
# -------------------------------------------
# NAME            |    MIN |    POS |    MAX
# shoulder        |    747 |    747 |   3004
# shoulder_shadow |    868 |    869 |   3139
# elbow           |   1028 |   3405 |   3408
# elbow_shadow    |    670 |   3045 |   3047
# wrist_angle     |   1237 |   2555 |   3587
# gripper         |    969 |   2080 |   2081
# INFO 2025-12-03 17:23:50 x/viperx.py:199 Calibration saved to /data/sichao/lerobot/.local/huggingface/lerobot/calibration/robots/viperx/left_follower.json
```

# Data Recording

```bash
lerobot-record \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
  --robot.max_relative_target=100.0 \
  --robot.cameras='{
    "top": {
        "type": "intelrealsense",
         "serial_number_or_name": "218622272943",
         "color_mode": "RGB",
         "use_depth": true,
         "rotation": 0,
         "warmup_s": 5,
         "width": 640,
         "height": 480,
         "fps": 30
     },
    "low": {
        "type": "intelrealsense",
        "serial_number_or_name": "218622276584",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "right": {
        "type": "intelrealsense",
        "serial_number_or_name": "230322276800",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    },
  }' \
  --teleop.type=widowx \
  --teleop.port=/dev/ttyDXL_leader_left \
  --teleop.id=left_leader \
  --teleop.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx \
  --display_data=true \
  --dataset.rename_map='{
    "observation.images.top":"observation.images.camera0",
    "observation.images.right":"observation.images.camera1",
    "observation.images.low":"observation.images.camera2"
  }' \
  --dataset.repo_id=lerobot/pick_sponge \
  --dataset.num_episodes=100 \
  --dataset.single_task="Pick the sponge into the box" \
  --dataset.push_to_hub=false \
  --resume=false

# Recommand to use this command to collect data, as the newest version of lerobot use lerobot-Dataset v3.0, which is quite different from formal version.
# --dataset.single_task is the language instruction for later training
# --resume=true means you can continue to collect data based on the data you have colleted;if first time to collect, --resume=false. 
```


## Tips:
#### Keyboard Controls During Recording
Control the data recording flow using keyboard shortcuts:

Press Right Arrow (→): Early stop the current episode or reset time and move to the next.

Press Left Arrow (←): Cancel the current episode and re-record it.

Press Escape (ESC): Immediately stop the session, encode videos, and upload the dataset.


<!-- More details can be found here:
https://huggingface.co/docs/lerobot/il_robots?record=Command#record-a-dataset


huggingface-cli login --token ${} --add-to-git-credential -->


# ACT Training 
```bash
lerobot-train \
  --dataset.repo_id=lerobot/pick_sponge \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_test \
  --job_name=act_aloha_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=lerobot/act_policy


  # This is a folder of /data/sichao/xinwang/lerobot
  # xingrui add viperx into the folder under /data/sichao/xinwang/lerobot/src/lerobot/robots/viperx
  # add widowx to folder under /data/sichao/xinwang/lerobot/src/lerobot/teleoperators/widowx

   Pytorch for GPU 5090
  # fine-tuning pi0 on GPU 5090, and it need higher version of pytorch, please think about 

  # https://github.com/facebookresearch/pytorch3d/issues/1962

pip uninstall -y torch torchvision torchaudio
pip uninstall -y xformers
  
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

```
# how to check if you are uisng ACT for policy training
```bash
# INPUT
(/data/sichao/conda_envs/lerobot) sichao@biorobpc25:/data/sichao/xinwang/lerobot$ lerobot-train   --dataset.repo_id=lerobot/pick_sponge   --policy.type=act   --output_dir=outputs/train/act_aloha_test   --job_name=act_aloha_test   --policy.device=cuda   --wandb.enable=true   --policy.repo_id=lerobot/act_policy

# OUTPUT: 
INFO 2025-11-27 17:13:08 ot_train.py:163 {'batch_size': 8,
 'checkpoint_path': None,
 'dataset': {'episodes': None,
             'image_transforms': {'enable': False,
                                  'max_num_transforms': 3,
                                  'random_order': False,
                                  'tfs': {'affine': {'kwargs': {'degrees': [-5.0,
                                                                            5.0],
                                                                'translate': [0.05,
                                                                              0.05]},
                                                     'type': 'RandomAffine',
                                                     'weight': 1.0},
                                          'brightness': {'kwargs': {'brightness': [0.8,
                                                                                   1.2]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'contrast': {'kwargs': {'contrast': [0.8,
                                                                               1.2]},
                                                       'type': 'ColorJitter',
                                                       'weight': 1.0},
                                          'hue': {'kwargs': {'hue': [-0.05,
                                                                     0.05]},
                                                  'type': 'ColorJitter',
                                                  'weight': 1.0},
                                          'saturation': {'kwargs': {'saturation': [0.5,
                                                                                   1.5]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'sharpness': {'kwargs': {'sharpness': [0.5,
                                                                                 1.5]},
                                                        'type': 'SharpnessJitter',
                                                        'weight': 1.0}}},
             'repo_id': 'lerobot/pick_sponge',
             'revision': None,
             'root': None,
             'streaming': False,
             'use_imagenet_stats': True,
             'video_backend': 'torchcodec'},
 'env': None,
 'eval': {'batch_size': 50, 'n_episodes': 50, 'use_async_envs': False},
 'eval_freq': 20000,
 'job_name': 'act_aloha_test',
 'log_freq': 200,
 'num_workers': 4,
 'optimizer': {'betas': [0.9, 0.999],
               'eps': 1e-08,
               'grad_clip_norm': 10.0,
               'lr': 1e-05,
               'type': 'adamw',
               'weight_decay': 0.0001},
 'output_dir': 'outputs/train/act_aloha_test',
 'policy': {'chunk_size': 100,
            'device': 'cuda',
            'dim_feedforward': 3200,
            'dim_model': 512,
            'dropout': 0.1,
            'feedforward_activation': 'relu',
            'input_features': {},
            'kl_weight': 10.0,
            'latent_dim': 32,
            'license': None,
            'n_action_steps': 100,
            'n_decoder_layers': 1,
            'n_encoder_layers': 4,
            'n_heads': 8,
            'n_obs_steps': 1,
            'n_vae_encoder_layers': 4,
            'normalization_mapping': {'ACTION': <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
                                      'STATE': <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
                                      'VISUAL': <NormalizationMode.MEAN_STD: 'MEAN_STD'>},
            'optimizer_lr': 1e-05,
            'optimizer_lr_backbone': 1e-05,
            'optimizer_weight_decay': 0.0001,
            'output_features': {},
            'pre_norm': False,
            'pretrained_backbone_weights': 'ResNet18_Weights.IMAGENET1K_V1',
            'pretrained_path': None,
            'private': None,
            'push_to_hub': True,
            'replace_final_stride_with_dilation': False,
            'repo_id': 'lerobot/act_policy',
            'tags': None,
            'temporal_ensemble_coeff': None,
            'type': 'act',  # this confirms you are using ACT
            'use_amp': False,
            'use_vae': True,
            'vision_backbone': 'resnet18'},
 'rename_map': {},
 'resume': False,
 'save_checkpoint': True,
 'save_freq': 20000,
 'scheduler': None,
 'seed': 1000,
 'steps': 100000,
 'use_policy_training_preset': True,
 'wandb': {'disable_artifact': False,
           'enable': True,
           'entity': None,
           'mode': None,
           'notes': None,
           'project': 'lerobot',
           'run_id': None}}
/data/sichao/conda_envs/lerobot/lib/python3.10/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/data/sichao/conda_envs/lerobot/lib/python3.10/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
INFO 2025-11-27 17:13:09 db_utils.py:102 Logs will be synced with wandb.
INFO 2025-11-27 17:13:09 db_utils.py:103 Track this run --> https://wandb.ai/cofstars-epfl/lerobot/runs/uxrd8t1p
INFO 2025-11-27 17:13:09 ot_train.py:183 Creating dataset
INFO 2025-11-27 17:13:09 ot_train.py:202 Creating policy
INFO 2025-11-27 17:13:10 ot_train.py:247 Creating optimizer and scheduler
INFO 2025-11-27 17:13:10 ot_train.py:259 Output dir: outputs/train/act_aloha_test
INFO 2025-11-27 17:13:10 ot_train.py:262 cfg.steps=100000 (100K)
INFO 2025-11-27 17:13:10 ot_train.py:263 dataset.num_frames=3042 (3K)
INFO 2025-11-27 17:13:10 ot_train.py:264 dataset.num_episodes=5
INFO 2025-11-27 17:13:10 ot_train.py:267 Effective batch size: 8 x 1 = 8
INFO 2025-11-27 17:13:10 ot_train.py:268 num_learnable_params=51603337 (52M)  # this also confirms ACT policy = 40–60M
INFO 2025-11-27 17:13:10 ot_train.py:269 num_total_params=51603337 (52M)
```


# Evaluating ACT
```bash
# Once training is complete, you can evaluate your ACT policy using the lerobot-record command with your trained policy. This will run inference and record evaluation episodes:

# ## offical command from LeRobot for evaluation
# lerobot-record \
#   --robot.type=so100_follower \
#   --robot.port=/dev/ttyACM0 \
#   --robot.id=my_robot \
#   --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
#   --display_data=true \
#   --dataset.repo_id=${HF_USER}/eval_act_your_dataset \
#   --dataset.num_episodes=10 \
#   --dataset.single_task="Your task description" \
#   --policy.path=${HF_USER}/act_policy


lerobot-record \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
  --robot.max_relative_target=100.0 \
  --robot.cameras='{
    "top": {
        "type": "intelrealsense",
         "serial_number_or_name": "218622272943",
         "color_mode": "RGB",
         "use_depth": true,
         "rotation": 0,
         "warmup_s": 5,
         "width": 640,
         "height": 480,
         "fps": 30
     },
    "low": {
        "type": "intelrealsense",
        "serial_number_or_name": "218622276584",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "right": {
        "type": "intelrealsense",
        "serial_number_or_name": "230322276800",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    }
  }' \
  --display_data=true \
  --policy.path=outputs/train/act_aloha_test \
  --dataset.repo_id=lerobot/pick_sponge_eval \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick the sponge into the box (Evaluation)" \
  --dataset.push_to_hub=false \
  --resume=false


# “--robot.calibration_dir” is required for ViperX; “--robot.type=so100_follower”; This robot type does not require encoder offsets, joint zero calibration, or DH parameter loading.
# “--robot.max_relative_target“ is needed for ViperX safety.The ViperX driver requires a max motion limit to avoid sudden jumps: "--robot.max_relative_target=100.0"; Without this, the robot will move unpredictably or LeRobot will crash.

# “--dataset.push_to_hub“ is optional (set to false for you); The official command uses: "--dataset.repo_id=${HF_USER}/eval_act_your_dataset"; By default, LeRobot pushes datasets to HuggingFace Hub; You do NOT want that, so we disable it: "--dataset.push_to_hub=false"

# “--resume=false“ ensures a clean new evaluation run; If an evaluation folder already exists, LeRobot may try to “resume” it; For safety, we say: “--resume=false”： This ensures a fresh evaluation dataset every time.
```

# Finetune SmolVLA on your data
```bash
# Use smolvla_base, our pretrained 450M model, and fine-tune it on your data. Training the model for 20k steps will roughly take ~4 hrs on a single A100 GPU. You should tune the number of steps based on performance and your use-case.


# ## offical command from LeRobot for fine-tuning
# lerobot-train \
#   --policy.path=lerobot/smolvla_base \
#   --dataset.repo_id=${HF_USER}/mydataset \
#   --batch_size=64 \
#   --steps=20000 \
#   --output_dir=outputs/train/my_smolvla \
#   --job_name=my_smolvla_training \
#   --policy.device=cuda \
#   --wandb.enable=true


# SmolVLA expects camera names camera0, camera1, camera2, ..., while your dataset still contains:
# observation.images.top
# observation.images.right
# observation.images.low
# Apply rename_map at training
# Add:
# --rename_map='{
#   "observation.images.top": "observation.images.camera0",
#   "observation.images.right": "observation.images.camera1",
#   "observation.images.low": "observation.images.camera2"
# }'


WANDB_IGNORE_GLOBS="losses_after_forward,losses_after_rm_padding" \
TOKENIZERS_PARALLELISM=false \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/pick_sponge \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{
    "observation.images.top": "observation.images.camera0",
    "observation.images.right": "observation.images.camera1",
    "observation.images.low": "observation.images.camera2"
  }' \
  --wandb.enable=true
```

# smolvla Evaluation and run inference 
```bash

lerobot-record \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
   --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
  --robot.max_relative_target=100.0 \
  --robot.cameras='{
    "top": {
        "type": "intelrealsense",
         "serial_number_or_name": "218622272943",
         "color_mode": "RGB",
         "use_depth": true,
         "rotation": 0,
         "warmup_s": 5,
         "width": 640,
         "height": 480,
         "fps": 30
     },
    "low": {
        "type": "intelrealsense",
        "serial_number_or_name": "218622276584",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,+
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "right": {
        "type": "intelrealsense",
        "serial_number_or_name": "230322276800",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    }
  }' \
  # --display_data=true \
  --dataset.episode_time_s=50 \ # sichao added
  --policy.path=outputs/train/smolvla_test \
  --dataset.repo_id=lerobot/pick_sponge \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick the sponge into the box (Evaluation)" \
  --dataset.push_to_hub=false \
  --resume=false


```

# Data Recording

```bash
lerobot-record \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/robots/viperx \
  --robot.max_relative_target=100.0 \
  --robot.cameras='{
    "top": {
        "type": "intelrealsense",
         "serial_number_or_name": "218622272943",
         "color_mode": "RGB",
         "use_depth": true,
         "rotation": 0,
         "warmup_s": 5,
         "width": 640,
         "height": 480,
         "fps": 30
     },
    "low": {
        "type": "intelrealsense",
        "serial_number_or_name": "218622276584",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "right": {
        "type": "intelrealsense",
        "serial_number_or_name": "230322276800",
        "color_mode": "RGB",
        "use_depth": true,
        "rotation": 0,
        "warmup_s": 5,
        "width": 640,
        "height": 480,
        "fps": 30
    }
  }' \
  --teleop.type=widowx \
  --teleop.port=/dev/ttyDXL_leader_left \
  --teleop.id=left_leader \
  --teleop.calibration_dir=/data/sichao/xinwang/lerobot/.local/huggingface/lerobot/calibration/teleoperators/widowx \
  --display_data=true \
  --dataset.rename_map='{
    "observation.images.top":"observation.images.camera0",
    "observation.images.low":"observation.images.camera2"
  }' \
  --dataset.repo_id=lerobot/pusht \
  --dataset.num_episodes=5 \
  --dataset.single_task="pushT" \
  --dataset.push_to_hub=false \
  --resume=false

# Diffusion-pusht training
```bash

# ## offical command from LeRobot for policy training
# python lerobot/scripts/train.py \
#     --output_dir=outputs/train/diffusion_pusht \
#     --policy.type=diffusion \
#     --dataset.repo_id=lerobot/pusht \
#     --seed=100000 \
#     --env.type=pusht \
#     --batch_size=64 \
#     --steps=200000 \
#     --eval_freq=25000 \
#     --save_freq=25000 \
#     --wandb.enable=true

lerobot-train \
  --dataset.repo_id=lerobot/pick_sponge \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_pusht \
  --job_name=diffusion_pusht_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=lerobot/diffusion_policy


```

# Diffusion-pusht evaluation
```bash
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/175000/pretrained_model \ 
    --output_dir=outputs/eval/diffusion_pusht/175000 \
    --env.type=pusht \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --device=cuda \
    --use_amp=false


```

# PI0.5 Training
```bash

lerobot-train \
    --dataset.repo_id=lerobot/pick_sponge \
    --policy.type=pi05 \
    --output_dir=outputs/train/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=lerobot/pi05_policy \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=4

# PI05 训练爆显存 的原因非常明确：
# 模型太大（≈4B 参数），启用 gradient checkpointing + compile_model + bfloat16 仍然不够，AdamW 在 step() 时需要额外 64 MB 做 foreach_sqrt，直接 OOM。
# 最有效解决方案：用 8-bit AdamW + AMP + 禁用 compile
# 你启用了 compile_model=true → 这个会让显存 增加（因为需要保存 graph + longer graph life）
# 你没有启用 use_amp=true → FP16 混合精度可以马上节约 40–50% 显存
# batch_size=4 —— 对 4B 模型来说太大

lerobot-train \
    --dataset.repo_id=lerobot/pick_sponge \
    --policy.type=pi05 \
    --output_dir=outputs/train/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=lerobot/pi05_policy \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.use_amp=true \
    --policy.dtype=bfloat16 \
    --optimizer.type=adamw_bnb_8bit \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=1


# 通过 --optimizer.optimizer_groups 加入 bnb 8-bit（推荐）
# LeRobot 支持 optimizer_groups，自定义优化器类型。
lerobot-train \
    --dataset.repo_id=lerobot/pick_sponge \
    --policy.type=pi05 \
    --output_dir=outputs/train/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=lerobot/pi05_policy \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.use_amp=true \
    --policy.dtype=bfloat16 \
    --optimizer.type=multi_adam \
    --optimizer.optimizer_groups="{'all_params': {'type': 'adamw_bnb_8bit', 'lr': 2.5e-5, 'weight_decay': 0.01}}" \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=1

```

# PI0.5 evaluation and ineference
```bash


```

# Training Pi0 
```bash
# For training π₀, you can use the standard LeRobot training script with the appropriate configuration:
# Install Pi0 dependencies by running:

pip install -e ".[pi]"
# [DEBUG] ValueError: An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues
pip install -e ".[pi]"
# Successfully installed lerobot-0.4.2 tokenizers-0.21.4 transformers-4.53.3

# ## offical command from LeRobot for Pi0 policy training
# python src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=your_dataset \
#     --policy.type=pi0 \
#     --output_dir=./outputs/pi0_training \
#     --job_name=pi0_training \
#     --policy.pretrained_path=lerobot/pi0_base \
#     --policy.repo_id=your_repo_id \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true \
#     --policy.dtype=bfloat16 \
#     --steps=3000 \
#     --policy.device=cuda \
#     --batch_size=32 

lerobot-train \
    --dataset.repo_id=lerobot/pick_sponge \
    --policy.type=pi0 \
    --output_dir=outputs/train/pi0_training \
    --job_name=pi0_training \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.repo_id=lerobot/pi0_policy \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 

# Disable compilation for PI0 completely: 
# add 
--policy.compile_model=false \
--policy.use_amp=true \       # recommended (BF16 or FP16 AMP)

lerobot-train \
  --dataset.repo_id=lerobot/pick_sponge \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_training \
  --job_name=pi0_training \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.repo_id=lerobot/pi0_policy \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.use_amp=true \
  --policy.dtype=bfloat16 \
  --steps=3000 \
  --policy.device=cuda \
  --batch_size=32


# Memory Breakdown
# Component	Approx VRAM
# Model weights (BF16)	8 GB
# AdamW exp_avg + exp_avg_sq	+16 GB
# Gradients	+8 GB
# Activations (after checkpointing)	~2–3 GB
# Total	34–36 GB

# Your GPU: 31.3 GB → impossible at batch size 32.

# Even batch size 8 is often borderline for 4B PI0.
# Guaranteed Fix: Reduce batch size
# Set:
--batch_size=4
# Try batch size 4 first (safe).
# If stable → try 6 or 8.

lerobot-train \
  --dataset.repo_id=lerobot/pick_sponge \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_training \
  --job_name=pi0_training \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.repo_id=lerobot/pi0_policy \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.use_amp=true \
  --policy.dtype=bfloat16 \
  --optimizer.type=adamw_bnb_8bit \
  --steps=3000 \
  --policy.device=cuda \
  --batch_size=4

# You cannot train PI0 (4B parameters) on a 31 GB GPU with AdamW — even at batch size = 1 — because AdamW optimizer states alone exceed your GPU memory.
# so you can Reduce PI0 Size (switch to pi05 or smolvla)
# PI0 = 4B
# PI05 = 2.7B
# smolvla = 1.5B

```


# Pi0 Evaluation
```bash


```
# lerobot's models
https://huggingface.co/lerobot/models
 
