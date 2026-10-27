import numpy as np

from egomimic.rldb.embodiment.piper_joint import PiperJoint


def test_joint_transform_shapes_and_binary_gripper():
    batch = {
        "action_joint": np.linspace(0.0, 1.0, 30 * 6, dtype=np.float32).reshape(30, 6),
        "action_gripper": np.concatenate(
            [np.zeros((15, 1), dtype=np.float32), np.ones((15, 1), dtype=np.float32)]
        ),
        "obs_joint": np.arange(6, dtype=np.float32),
        "obs_gripper": np.asarray([0.4], dtype=np.float32),
    }
    output = batch
    for transform in PiperJoint.get_transform_list():
        output = transform.transform(output)
    assert tuple(output[PiperJoint.ACTION_KEY].shape) == (100, 7)
    assert tuple(output[PiperJoint.PROPRIO_KEY].shape) == (7,)
    assert set(output[PiperJoint.ACTION_KEY][:, 6].numpy().tolist()) == {0.0, 1.0}
