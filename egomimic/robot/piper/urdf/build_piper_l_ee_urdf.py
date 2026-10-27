"""Regenerates piper_l_ee.urdf from piper_l_description.urdf.

Both source files are pulled verbatim from agilexrobotics/agx_arm_urdf
(piper_l/urdf/). This script grafts the fixed link6 -> flange_link ->
gripper_base transform (copied from piper_l_with_gripper_description.xacro's
corresponding <link>/<joint> elements) onto the 6-revolute-joint arm chain,
producing one flat, mesh-free, xacro-free URDF that FK libraries (this repo
uses pytorch_kinematics, see ../piper_kinematics.py) can load directly.

"gripper_base" -- the fixed frame at the root of the two-finger gripper, before
the jaws -- is used as the end-effector frame. The xacro's two prismatic finger
joints (gripper_joint1/2, both `mimic`-driven by the single `gripper` joint) are
intentionally left out: they move the jaws, not gripper_base, and mimic joints
add complexity that FK-to-gripper_base doesn't need.

Re-run this if either source URDF is refreshed from upstream:
    python build_piper_l_ee_urdf.py
"""

import xml.etree.ElementTree as ET

FLANGE_LINK = """
<link name="flange_link">
  <inertial>
    <origin xyz="1.151691e-03 0.782428e-03 4.798051e-03" rpy="0 0 0"/>
    <mass value="0.04"/>
    <inertia ixx="9.935389e-06" ixy="0.0" ixz="0.0" iyy="10.508003e-06" iyz="0.0" izz="19.652873e-06"/>
  </inertial>
</link>
"""

FLANGE_JOINT = """
<joint name="flange_joint" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="link6"/>
  <child link="flange_link"/>
</joint>
"""

GRIPPER_BASE_LINK = """
<link name="gripper_base">
  <inertial>
    <origin xyz="-0.000183807162235591 8.05033155577911E-05 0.0321436689908876" rpy="0 0 0"/>
    <mass value="0.45"/>
    <inertia ixx="0.00092934" ixy="0.00000034" ixz="-0.00000738" iyy="0.00071447" iyz="0.00000005" izz="0.00039442"/>
  </inertial>
</link>
"""

GRIPPER_BASE_JOINT = """
<joint name="gripper_base_joint" type="fixed">
  <origin xyz="0 0 0.0045" rpy="0 0 0"/>
  <parent link="flange_link"/>
  <child link="gripper_base"/>
</joint>
"""


def main():
    tree = ET.parse("piper_l_description.urdf")
    root = tree.getroot()
    for xml_str in (FLANGE_LINK, FLANGE_JOINT, GRIPPER_BASE_LINK, GRIPPER_BASE_JOINT):
        root.append(ET.fromstring(xml_str))
    tree.write("piper_l_ee.urdf", xml_declaration=True, encoding="utf-8")
    print("wrote piper_l_ee.urdf")


if __name__ == "__main__":
    main()
