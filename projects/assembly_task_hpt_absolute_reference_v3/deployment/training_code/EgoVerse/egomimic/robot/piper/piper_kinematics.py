"""Forward kinematics for the AgileX PIPER-L follower arm.

Why this doesn't use `egomimic.robot.kinematics.MinkKinematicsSolver` (the pattern
`EvaMinkKinematicsSolver` follows): that solver needs `mink`, which isn't installed
in this env, and Eva never actually needed offline FK in the first place -- Eva's
`obs_ee_pose` comes straight from the robot's own live pose feedback, recorded at
teleop time. PIPER only exposes joint angles (no onboard EE-pose readout), so
turning a whole recorded dataset into `obs_ee_pose` means running FK over every
frame after the fact -- exactly the case `pytorch_kinematics` (already installed,
batched, no mesh files required) is built for.

URDF provenance -- REVISED, see below for why: the arm chain (joints 1-6) is now
generated programmatically from AgileX's own official DH parameters
(`piper_sdk.kinematics.piper_fk.C_PiperForwardKinematics`, `dh_is_offset=1` -- the
default `C_PiperInterface_V2()` and this codebase's own deployment examples use),
not from agilexrobotics/agx_arm_urdf's URDF -- see `urdf/piper_l_dh.urdf` and
`build_piper_l_dh_urdf.py` in this directory. The `flange_link -> gripper_base`
fixed transform is still grafted on from piper_l/urdf/piper_l_with_gripper_
description.xacro (unchanged, copied verbatim from the old `piper_l_ee.urdf`) --
that offset was never implicated in the bug below. "gripper_base" -- the fixed
frame at the root of the two-finger gripper, before the jaws -- is what this
module calls the end-effector frame. The two prismatic finger joints are
deliberately not included: they don't affect gripper_base's pose, only the jaws'
own.

WHY THIS CHANGED -- a real, confirmed bug, not a style preference: the previous
`agx_arm_urdf`-derived URDF (`urdf/piper_l_ee.urdf`, still present for reference,
no longer the default) disagreed with AgileX's own official FK by 20-100mm,
growing with distance from the home pose. This was confirmed three independent
ways: (1) synthetic joint angles compared directly against
`piper_sdk...CalFK` with no camera involved at all; (2) the same comparison using
real recorded joint angles from actual episodes; (3) reprojecting both FKs'
predicted gripper position onto the real camera frame they were recorded from --
the SDK's point landed on the physical gripper, the old URDF's did not. The
earlier "URDF's joint5 limit lines up almost exactly with the observed joint_4
raw-state extremes" evidence (still true, see git history) turned out to be a
correct-but-insufficient check: it validated joint order/correspondence, not the
actual link geometry (lengths/origins) the other joints depend on -- a
symmetric-range joint like joint5 can have its limits "coincidentally" line up
while an upstream link length is still wrong. The new `piper_l_dh.urdf` is built
directly from the DH table with no hand-transcription (every joint origin's
rpy was round-trip verified before being written), and its FK was independently
verified to match `CalFK` to floating-point precision (<0.001mm) over 200 random
full-range configs -- see `check_pipeline.py`'s "FK matches AgileX official SDK"
check, which is a regression test for exactly this bug class.

CONSEQUENCE FOR EXISTING DATA: any Zarr episode converted before this fix (e.g.
`piper_zarr_teddy_bear`, and an initial `piper_zarr_teddy_0912` conversion done
the same session this was found) has `obs_ee_pose` computed with the old, wrong
FK and should be re-converted. `Piper.EXTRINSICS` was also calibrated using the
old FK as ground truth (`calibrate_piper_extrinsics.py` calls `kin.fk()`), so it
likely absorbed some of this bias for its specific calibration-sample pose range
-- also due for re-calibration, ideally after re-converting data, though this
needs physical access to the robot/camera rig this session doesn't have.

Raw-unit reference (from lerobot's `piper_sdk_interface.py`, itself sourced from
piper_sdk's message definitions):
  - `observation.state` in the recorded dataset: raw joint feedback, 0.001deg per
    count for joints 0-5, 0.001mm per count for the gripper (joint_6). Absolute,
    unit-unambiguous -- safe to feed straight through FK after a deg->rad convert.
  - `action` in the recorded dataset: PIPER's `send_action` always normalizes to
    [-100, 100] (joints) / [0, 100] (gripper) using *this specific unit's*
    calibrated per-joint min/max angle limits (`piper.min_pos`/`max_pos`), which
    are NOT stored anywhere in the dataset (queried live from the arm, or from a
    `PiperConfig.joint_limits_deg` override, at recording time). There is no way
    to recover real joint angles from the recorded `action` column alone -- so
    this module (and the Piper embodiment built on it) deliberately never runs FK
    on `action`. Cartesian *targets* are instead built the same way EgoVerse's
    `Human` embodiment already handles data with no independently-calibrated
    commanded pose: pose targets are built by stacking future `obs_ee_pose`
    frames, while gripper targets use binary intent recovered from the recorded
    gripper command (see `egomimic/rldb/embodiment/piper.py`).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    import pytorch_kinematics as pk
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch_kinematics is required for Piper FK. `pip install pytorch-kinematics`."
    ) from e


_URDF_DIR = os.path.join(os.path.dirname(__file__), "urdf")
DEFAULT_URDF_PATH = os.path.join(_URDF_DIR, "piper_l_dh.urdf")
# The old agx_arm_urdf-derived file, kept only for reference/history -- see this
# module's docstring for why it's no longer the default (confirmed ~20-100mm off
# vs AgileX's own official FK).
_LEGACY_URDF_PATH = os.path.join(_URDF_DIR, "piper_l_ee.urdf")

# PIPER-L two-finger gripper, "Opening Range: 0-100mm" per AgileX's datasheet
# (standard, non-L PIPER is 70mm -- do not reuse this constant for that hardware).
GRIPPER_STROKE_MM = 100.0


def raw_joint_to_rad(raw_deg001: np.ndarray) -> np.ndarray:
    """Raw 0.001deg joint feedback (`observation.state` / `action` columns,
    joints 0-5) -> radians."""
    return np.asarray(raw_deg001, dtype=np.float64) * 1e-3 * np.pi / 180.0


def raw_gripper_to_frac(raw_mm001: np.ndarray, stroke_mm: float = GRIPPER_STROKE_MM) -> np.ndarray:
    """Raw 0.001mm gripper feedback (`observation.state` / `action` column
    joint_6) -> normalized [0, 1] aperture (0 = fully closed), matching
    CONTRIBUTING_DATA.md's `obs_gripper` convention."""
    frac = np.asarray(raw_mm001, dtype=np.float64) * 1e-3 / stroke_mm
    return np.clip(frac, 0.0, 1.0)


class PiperKinematicsSolver:
    """Forward *and* inverse kinematics for the PIPER-L 6-DOF arm chain, via
    `pytorch_kinematics`. FK is used for Zarr conversion (offline, batched);
    IK is used for real-hardware rollout (predicted cartesian action -> joint
    command), via `pytorch_kinematics.ik.PseudoInverseIK` -- a damped
    least-squares Jacobian solver with joint-limit enforcement and
    multi-restart, built on this same URDF chain.
    """

    NUM_JOINTS = 6

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF_PATH,
        end_link_name: str = "gripper_base",
        root_link_name: str = "base_link",
        device: str = "cpu",
        joint_limits_rad: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """
        joint_limits_rad: optional (low, high), each (NUM_JOINTS,) in radians,
            overriding the URDF's <limit> values for IK's retry-clamping and
            joint-limit enforcement (does NOT affect `fk`, which never
            consults limits). Use this unit's REAL calibrated joint limits
            (see PiperConfig.joint_limits_deg / find_real_joint_limits.py) --
            the generic agilex reference URDF's limits are idealized/nominal
            (e.g. exactly [0, 180] deg for joint2) and can be a couple degrees
            narrower than what a real unit's mechanical stops actually allow.
            Left as None, `ik()` raises by default rather than silently
            solving against the URDF's narrower/wrong box -- pass
            joint_limits_rad here (preferred), or ik()'s own
            allow_nominal_urdf_limits=True for a call that provably never
            reaches a real arm. Before this guard existed, an un-overridden
            solver would silently clamp any retry -- including the seed
            itself -- into that gap, which can make even a trivial "hold
            current pose" IK solve fail to converge if the arm happens to be
            resting right at the boundary. Discovered via a real hardware
            IK-failure investigation where the arm's actual (valid) resting
            pose had joint2=-2.06 deg, joint3=1.98 deg -- both outside the
            URDF's [0,180]/[-170,0] deg limits but within this unit's real
            calibrated range.
        """
        with open(urdf_path, "rb") as f:
            urdf_bytes = f.read()
        self.chain = pk.build_serial_chain_from_urdf(
            urdf_bytes, end_link_name=end_link_name, root_link_name=root_link_name
        )
        self.device = device
        self.chain = self.chain.to(device=device, dtype=torch.float32)
        joint_names = self.chain.get_joint_parameter_names()
        assert joint_names == [f"joint{i + 1}" for i in range(self.NUM_JOINTS)], (
            f"Unexpected joint order/names from URDF: {joint_names}"
        )
        # Tracked so `ik()` can refuse to run on real hardware with the URDF's
        # nominal limits by accident -- see this file's real-unit vs. URDF
        # comparison in `piper_extrinsics_result.txt`'s sibling investigation
        # (joint2/joint6 especially: several to ~50deg off). `fk()` never
        # consults this, so it's irrelevant to FK-only callers (Zarr conversion,
        # cmd_ee_pose recovery) regardless of which way this flag ends up.
        self._limits_are_calibrated = joint_limits_rad is not None
        # Local-continuity IK temporarily narrows the chain limits to a trust
        # region around the measured/previous joint state.  The deployment
        # path is single-threaded, but protect the temporary mutation so an
        # accidental concurrent diagnostic cannot observe narrowed limits.
        self._ik_lock = threading.RLock()
        self.last_ik_diagnostics: dict | None = None
        if joint_limits_rad is not None:
            low, high = joint_limits_rad
            low = torch.as_tensor(low, dtype=torch.float32, device=device)
            high = torch.as_tensor(high, dtype=torch.float32, device=device)
            if low.shape != (self.NUM_JOINTS,) or high.shape != (self.NUM_JOINTS,):
                raise ValueError(
                    f"joint_limits_rad low/high must each be ({self.NUM_JOINTS},), "
                    f"got {low.shape}/{high.shape}"
                )
            if torch.any(high <= low):
                raise ValueError(f"joint_limits_rad has a non-increasing pair: low={low}, high={high}")
            self.chain.low = low
            self.chain.high = high

    def fk(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """
        Args:
            joint_angles_rad: (..., 6) joint angles in radians (joint1..joint6).
                Any leading batch shape (T,), (B, T), etc. is supported.

        Returns:
            (..., 7) end-effector pose as XYZWXYZ (matching CONTRIBUTING_DATA.md's
            pose convention: [tx, ty, tz, qw, qx, qy, qz]), in the arm base frame.
        """
        q = np.asarray(joint_angles_rad, dtype=np.float32)
        assert q.shape[-1] == self.NUM_JOINTS, f"expected last dim {self.NUM_JOINTS}, got {q.shape}"
        lead_shape = q.shape[:-1]
        q_flat = torch.from_numpy(q.reshape(-1, self.NUM_JOINTS)).to(self.device)

        with torch.no_grad():
            tf = self.chain.forward_kinematics(q_flat)
            mat = tf.get_matrix().cpu().numpy()  # (N, 4, 4)

        pos = mat[:, :3, 3]
        quat_xyzw = R.from_matrix(mat[:, :3, :3]).as_quat()  # (N, 4), [x, y, z, w]
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        pose = np.concatenate([pos, quat_wxyz], axis=-1)  # (N, 7) XYZWXYZ
        return pose.reshape(*lead_shape, 7)

    def ik(
        self,
        target_pose_xyzwxyz: np.ndarray,
        seed_joint_angles_rad: np.ndarray | None = None,
        num_retries: int = 32,
        pos_tolerance: float = 1e-3,
        rot_tolerance: float = 2e-2,
        max_iterations: int = 100,
        seed_noise_std: float = 0.2,
        allow_nominal_urdf_limits: bool = False,
        max_normalized_joint_step: float | None = None,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve IK for a single target gripper_base pose.

        Args:
            target_pose_xyzwxyz: (7,) target pose, XYZWXYZ, base_link frame
                (the same convention `fk` returns -- i.e. whatever a trained
                policy predicts, once unnormalized, is directly usable here).
            allow_nominal_urdf_limits: must be explicitly set True to run IK
                without having passed real `joint_limits_rad` at construction.
                Default False raises instead of silently solving against the
                URDF's idealized/nominal limits -- measured on this rig's
                actual `my_piper_follower` unit, those are off from the real
                calibrated range by several degrees on most joints and by
                ~50deg on joint6 (see this class's `__init__` docstring). A
                solve against the wrong box doesn't just give a slightly-off
                answer -- it can make a trivially valid "hold current pose"
                solve fail to converge if the arm's real resting angle falls
                in the gap, exactly as happened once already on real
                hardware. Only set this True for an FK-adjacent sanity check
                that never reaches a real arm (`fk()` itself is unaffected by
                this flag either way -- it never consults limits at all).
            seed_joint_angles_rad: (6,) current/previous joint config, if
                known. Retries are then sampled as small perturbations around
                the seed (plus the seed itself) rather than uniformly at
                random across the whole joint-limit box. This matters for
                real-time control: an IK solver has no notion of "the
                solution closest to where the arm already is" unless you bias
                its search that way, and an unbiased solve can return a
                valid-but-wildly-different configuration (e.g. an elbow-flip)
                that would jerk the arm. Leave `None` only for a first solve
                with no prior joint state to seed from.
            num_retries: random-restart attempts (or perturbed-around-seed
                attempts, if seeding).
            max_normalized_joint_step: optional local IK trust-region radius
                in the deployment [-100, 100] joint command space.  When a
                seed is provided, every IK iteration and returned candidate
                is constrained to this distance per joint from that seed.
                This prevents a numerically exact target from being reached
                by silently switching to a distant elbow/base/wrist branch.
            position_weight/orientation_weight: task-space weights passed to
                the damped least-squares solver.  They influence convergence
                but do not relax ``pos_tolerance`` or ``rot_tolerance``.
        Returns:
            (joint_angles_rad (6,), converged, pos_error_m). `converged`
            False means no retry satisfied both tolerances -- the returned
            angles are still the closest attempt found (lowest combined
            error), but the caller should NOT trust them blindly; see this
            method's docstring in the calling rollout code for what to do
            (typically: skip the command / hold position / drop the affected
            demonstration frame, and log it loudly, rather than execute
            anything on the real arm on an unconverged solve).
        """
        if not self._limits_are_calibrated and not allow_nominal_urdf_limits:
            raise RuntimeError(
                "PiperKinematicsSolver.ik() called without real calibrated joint "
                "limits (construct with joint_limits_rad=..., see "
                "PiperConfig.joint_limits_deg / find_real_joint_limits.py for this "
                "unit's real values -- do NOT reuse another unit's calibration "
                "file). The URDF's nominal limits measured off this rig's real "
                "my_piper_follower calibration by several degrees on most joints "
                "and ~50deg on joint6; solving against them can fail to converge "
                "on even a trivial hold-pose target. Pass "
                "allow_nominal_urdf_limits=True only if this call never reaches "
                "a real arm."
            )
        if max_normalized_joint_step is not None:
            if seed_joint_angles_rad is None:
                raise ValueError(
                    "max_normalized_joint_step requires seed_joint_angles_rad"
                )
            if not 0.0 < max_normalized_joint_step <= 200.0:
                raise ValueError(
                    "max_normalized_joint_step must be in (0, 200]"
                )
        if position_weight <= 0.0 or orientation_weight <= 0.0:
            raise ValueError("IK position/orientation weights must be positive")
        pos = np.asarray(target_pose_xyzwxyz[:3], dtype=np.float32)
        quat_wxyz = np.asarray(target_pose_xyzwxyz[3:7], dtype=np.float32)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        rot_mat = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = pos
        target = pk.Transform3d(
            matrix=torch.from_numpy(mat).unsqueeze(0).to(self.device)
        )

        global_low = self.chain.low.clone()
        global_high = self.chain.high.clone()

        if seed_joint_angles_rad is not None:
            seed = torch.as_tensor(
                seed_joint_angles_rad, dtype=torch.float32, device=self.device
            )
            if max_normalized_joint_step is None:
                solve_low = global_low
                solve_high = global_high
            else:
                radius = (global_high - global_low) * (
                    float(max_normalized_joint_step) / 200.0
                )
                solve_low = torch.maximum(global_low, seed - radius)
                solve_high = torch.minimum(global_high, seed + radius)
            joint_limits = torch.stack([solve_low, solve_high], dim=1)
            noise = torch.randn(num_retries, self.NUM_JOINTS, device=self.device) * seed_noise_std
            retry_configs = seed.unsqueeze(0) + noise
            retry_configs[0] = seed  # always try the unperturbed seed itself
            retry_configs = torch.clamp(retry_configs, joint_limits[:, 0], joint_limits[:, 1])
            ik_solver = pk.PseudoInverseIK(
                self.chain,
                retry_configs=retry_configs,
                pos_tolerance=pos_tolerance,
                rot_tolerance=rot_tolerance,
                max_iterations=max_iterations,
                joint_limits=joint_limits,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                clamp_to_limits=max_normalized_joint_step is not None,
                early_stopping_no_improvement=(
                    "all" if max_normalized_joint_step is not None else "any"
                ),
                early_stopping_no_improvement_patience=(
                    5 if max_normalized_joint_step is not None else 2
                ),
            )
        else:
            solve_low = global_low
            solve_high = global_high
            joint_limits = torch.stack([solve_low, solve_high], dim=1)
            ik_solver = pk.PseudoInverseIK(
                self.chain,
                num_retries=num_retries,
                pos_tolerance=pos_tolerance,
                rot_tolerance=rot_tolerance,
                max_iterations=max_iterations,
                joint_limits=joint_limits,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )

        with self._ik_lock:
            try:
                # PseudoInverseIK's projected limit refinement consults the
                # chain limits directly.  Narrow them for this solve so the
                # optimizer itself stays on the local branch, then restore the
                # calibrated global limits immediately.
                self.chain.low = solve_low
                self.chain.high = solve_high
                with torch.no_grad():
                    sol = ik_solver.solve(target)
            finally:
                self.chain.low = global_low
                self.chain.high = global_high

        solutions = sol.solutions[0]  # (num_retries, 6)
        converged = sol.converged[0]  # (num_retries,)
        err_pos = sol.err_pos[0]
        err_rot = sol.err_rot[0]
        total_err = err_pos + err_rot

        if converged.any():
            candidates = torch.nonzero(converged).squeeze(-1)
            if seed_joint_angles_rad is not None:
                seed_t = torch.as_tensor(seed_joint_angles_rad, dtype=torch.float32, device=self.device)
                # Compare candidates in each joint's normalized command
                # space, not raw radians. A one-radian change in a narrow
                # joint is much more consequential than the same raw change
                # in a wide-range joint; range-normalized distance matches
                # the [-100,100] actuator frame used by deployment and makes
                # the selected IK branch the closest one to measured state.
                joint_span = (global_high - global_low).clamp_min(1e-6)
                normalized_delta = (solutions[candidates] - seed_t) / joint_span
                dist = normalized_delta.norm(dim=-1)
                best = candidates[dist.argmin()]
            else:
                best = candidates[total_err[candidates].argmin()]
            is_converged = True
        else:
            best = total_err.argmin()
            is_converged = False

        best_q = solutions[best].cpu().numpy()
        best_pos_err = float(err_pos[best].cpu().item())
        best_rot_err = float(err_rot[best].cpu().item())
        seed_delta_normalized = None
        if seed_joint_angles_rad is not None:
            seed_np = np.asarray(seed_joint_angles_rad, dtype=np.float64)
            span_np = (global_high - global_low).cpu().numpy().astype(np.float64)
            seed_delta_normalized = (
                (best_q.astype(np.float64) - seed_np) / span_np * 200.0
            )
        self.last_ik_diagnostics = {
            "converged": bool(is_converged),
            "position_error_m": best_pos_err,
            "rotation_error_rad": best_rot_err,
            "converged_candidates": int(converged.sum().item()),
            "selected_retry": int(best.item()),
            "max_normalized_joint_step": (
                None
                if max_normalized_joint_step is None
                else float(max_normalized_joint_step)
            ),
            "seed_joint_delta_normalized": (
                None
                if seed_delta_normalized is None
                else seed_delta_normalized.tolist()
            ),
            "seed_joint_rms_normalized": (
                None
                if seed_delta_normalized is None
                else float(np.sqrt(np.mean(seed_delta_normalized**2)))
            ),
            "seed_joint_max_abs_normalized": (
                None
                if seed_delta_normalized is None
                else float(np.max(np.abs(seed_delta_normalized)))
            ),
        }
        return best_q, is_converged, best_pos_err


if __name__ == "__main__":
    # Quick sanity check.
    solver = PiperKinematicsSolver()
    q0 = np.zeros((6,), dtype=np.float32)
    print("zero pose (gripper_base, base_link frame), XYZWXYZ:", solver.fk(q0))

    q_batch = np.stack([np.zeros(6), np.array([0.1, 0.5, -0.5, 0.0, 0.3, 0.0])], axis=0)
    print("batched:", solver.fk(q_batch).shape)
