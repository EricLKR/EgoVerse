import numpy as np

from egomimic.rldb.embodiment.piper_human_joint import PiperHumanJoint


def test_joint_human_transform_has_shared_pose_and_binary_robot_action():
    pose = np.zeros((30, 7), dtype=np.float32)
    pose[:, 0] = np.linspace(0.10, 0.20, 30)
    pose[:, 1] = np.linspace(-0.05, 0.05, 30)
    pose[:, 2] = 0.25
    pose[:, 3] = 1.0  # xyz + wxyz identity quaternion
    batch = {
        "action_ee_pose": pose,
        "action_joint": np.linspace(-0.5, 0.5, 30 * 6, dtype=np.float32).reshape(30, 6),
        "action_gripper": np.concatenate(
            [np.zeros((15, 1), dtype=np.float32), np.ones((15, 1), dtype=np.float32)]
        ),
        "obs_joint": np.arange(6, dtype=np.float32) / 10.0,
        "obs_gripper": np.asarray([0.4], dtype=np.float32),
    }
    output = batch
    for transform in PiperHumanJoint.get_transform_list():
        output = transform.transform(output)

    assert tuple(output[PiperHumanJoint.SHARED_ACTION_KEY].shape) == (100, 9)
    assert tuple(output[PiperHumanJoint.ACTION_KEY].shape) == (100, 7)
    assert tuple(output[PiperHumanJoint.PROPRIO_KEY].shape) == (7,)
    assert set(output[PiperHumanJoint.ACTION_KEY][:, 6].numpy().tolist()) == {0.0, 1.0}
    assert "action_ee_pose" not in output
    assert "extrinsics_pose" not in output
