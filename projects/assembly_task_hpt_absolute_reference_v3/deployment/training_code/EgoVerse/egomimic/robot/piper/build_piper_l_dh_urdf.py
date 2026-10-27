#!/usr/bin/env python
"""Generates urdf/piper_l_dh.urdf directly from AgileX's own official DH parameters
(piper_sdk.kinematics.piper_fk.C_PiperForwardKinematics, dh_is_offset=1) -- see
piper_kinematics.py's module docstring for why this replaced the previous
agx_arm_urdf-derived URDF (confirmed ~20-100mm off vs the vendor's own FK).

Method: AgileX uses the Modified/Craig DH convention -- confirmed empirically (not
assumed from memory) by testing candidate matrix compositions against
C_PiperForwardKinematics._C_PiperForwardKinematics__LinkTransformtion's actual output:

    T_i = Rot_x(alpha_i) @ Trans_x(a_i) @ Rot_z(theta_i) @ Trans_z(d_i)

where theta_i = raw_joint_angle_i + theta_offset_i (theta_offset_i is AgileX's own
per-joint DH constant, e.g. joint3's -102.78deg -- NOT a calibration value, it's part
of the vendor's kinematic model and is the same for every physical unit). Since
Rot_z(a + b) = Rot_z(a) @ Rot_z(b) for same-axis rotations, this factors into a joint
URDF can express directly:

    T_i = [Rot_x(alpha_i) @ Trans_x(a_i) @ Rot_z(theta_offset_i)]   <- joint's fixed <origin>
          @ Rot_z(raw_joint_angle_i)                                 <- joint's own revolute rotation
          @ Trans_z(d_i)                                             <- a further FIXED joint's <origin>,
                                                                          needed because URDF has no
                                                                          "translate after rotating" slot
                                                                          within a single joint element

So each DH row becomes two URDF joints: "jointN" (revolute, the real joint) followed by
"jointN_post" (fixed, translates by d_N). Every generated <origin> rpy is round-trip
verified (matrix -> euler -> matrix, asserted <1e-9 error) before being written, and the
resulting URDF's FK is verified elsewhere (see check_pipeline.py's "FK matches AgileX
official SDK" check) to match CalFK to floating-point precision over random configs --
run that check after regenerating this file, not just this script's own assertions.

The flange_link/gripper_base tail (fixed, ~4.5mm) is copied verbatim from the old
urdf/piper_l_ee.urdf -- never implicated in the discrepancy, no need to regenerate it.

Usage:
    python build_piper_l_dh_urdf.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation as R

# AgileX's own DH parameters (piper_sdk.kinematics.piper_fk.C_PiperForwardKinematics,
# dh_is_offset=1 -- the default C_PiperInterface_V2() and this codebase's deployment
# examples use). Transcribed directly from that class's __init__, not re-derived.
_A_MM = [0, 0, 285.03, -21.98, 0, 0]
_ALPHA_RAD = [0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2]
_THETA_OFFSET_RAD = [0, -172.22 / 180 * np.pi, -102.78 / 180 * np.pi, 0, 0, 0]
_D_MM = [123, 0, 0, 250.75, 0, 91]

_OUT_PATH = os.path.join(os.path.dirname(__file__), "urdf", "piper_l_dh.urdf")


def _rotx(a: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = R.from_euler("x", a).as_matrix()
    return m


def _transx(a: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 3] = a
    return m


def _rotz(t: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = R.from_euler("z", t).as_matrix()
    return m


def build_urdf() -> str:
    joints_xml = []
    for i in range(6):
        t_pre = _rotx(_ALPHA_RAD[i]) @ _transx(_A_MM[i]) @ _rotz(_THETA_OFFSET_RAD[i])
        xyz_m = t_pre[:3, 3] / 1000.0  # mm -> m
        rpy = R.from_matrix(t_pre[:3, :3]).as_euler("xyz", degrees=False)

        recon = np.eye(4)
        recon[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        recon[:3, 3] = t_pre[:3, 3]
        err = np.max(np.abs(recon - t_pre))
        assert err < 1e-9, f"joint{i + 1}: origin round-trip error {err:.2e}"

        parent = "base_link" if i == 0 else f"link{i}_dh"
        joints_xml.append(f"""
  <link name="link{i + 1}_pre"/>
  <joint name="joint{i + 1}" type="revolute">
    <parent link="{parent}"/>
    <child link="link{i + 1}_pre"/>
    <origin xyz="{xyz_m[0]:.9f} {xyz_m[1]:.9f} {xyz_m[2]:.9f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="5"/>
  </joint>
  <link name="link{i + 1}_dh"/>
  <joint name="joint{i + 1}_post" type="fixed">
    <parent link="link{i + 1}_pre"/>
    <child link="link{i + 1}_dh"/>
    <origin xyz="0 0 {_D_MM[i] / 1000.0:.9f}" rpy="0 0 0"/>
  </joint>""")

    comment = (
        "DH derived Piper-L arm chain; see build_piper_l_dh_urdf.py (this file's "
        "generator) and piper_kinematics.py's module docstring for full provenance "
        "and the bug this replaced. link6_dh is the same physical frame piper_fk.py "
        "calls joint6/R06. The flange_link/gripper_base tail is copied unchanged "
        "from piper_l_ee.urdf (a small, about 4.5mm fixed offset never implicated "
        "in the discrepancy)."
    )
    assert "--" not in comment

    urdf = f"""<?xml version="1.0"?>
<!-- {comment} -->
<robot name="piper_l_dh">
  <link name="world" />
  <joint name="world_to_base_link" type="fixed">
    <parent link="world" />
    <child link="base_link" />
  </joint>
  <link name="base_link"/>
{"".join(joints_xml)}
<link name="flange_link">
</link><joint name="flange_joint" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0" />
  <parent link="link6_dh" />
  <child link="flange_link" />
</joint><link name="gripper_base">
</link><joint name="gripper_base_joint" type="fixed">
  <origin xyz="0 0 0.0045" rpy="0 0 0" />
  <parent link="flange_link" />
  <child link="gripper_base" />
</joint></robot>
"""
    assert urdf.count("<!--") == 1 and urdf.count("-->") == 1
    return urdf


if __name__ == "__main__":
    content = build_urdf()
    with open(_OUT_PATH, "w") as f:
        f.write(content)
    print(f"wrote {_OUT_PATH} ({len(content)} bytes)")
