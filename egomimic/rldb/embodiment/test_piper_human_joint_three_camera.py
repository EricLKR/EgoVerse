from egomimic.rldb.embodiment.pi05_expert_action_v3_three_camera import (
    PiperExpertActionV3ThreeCamera,
)
from egomimic.rldb.embodiment.piper_human_joint_three_camera import (
    PiperHumanJointThreeCamera,
)


def _assert_three_camera_mapping(mapping):
    assert mapping["observations.images.side_img"] == {
        "key_type": "camera_keys",
        "zarr_key": "images.side_1",
    }
    assert mapping["observations.images.wrist_img"]["zarr_key"] == "images.wrist_1"
    assert mapping["observations.images.front_img_1"]["zarr_key"] == "images.front_1"


def test_hpt_three_camera_keymap_preserves_action_contract():
    mapping = PiperHumanJointThreeCamera._get_keymap("joint_cartesian_camframe")
    _assert_three_camera_mapping(mapping)
    assert mapping["action_joint"]["horizon"] == 30
    assert mapping["action_gripper"]["horizon"] == 30
    assert mapping["action_ee_pose"]["horizon"] == 30


def test_pi05_three_camera_contract_is_native_30_steps():
    mapping = PiperExpertActionV3ThreeCamera._get_keymap("joint_cartesian_camframe")
    _assert_three_camera_mapping(mapping)
    assert PiperExpertActionV3ThreeCamera.ACTION_HORIZON == 30
    assert PiperExpertActionV3ThreeCamera.CHUNK_LENGTH == 30
