"""Client-side bridge to the persistent S7 inference worker (egoverse_jepa/egomimic/scripts/experiments/
egoverse_a_jepa_n4_2d_6_s7_worker.py), mirroring egoverse_jepa's own preprocess/SAM3Ops.py pattern exactly:
a long-running subprocess launched with a DIFFERENT conda env's python interpreter, JSON-lines over
stdin/stdout, large data (frames) passed via temp .npy file paths rather than inline.

This file NEVER imports anything from egoverse_jepa's `egomimic` fork -- that fork's `egomimic.models.
hpt_nets` has genuinely diverged from this deployment env's own `egomimic` (the one with real Piper robot
control), confirmed directly (165 lines of real difference, not cosmetic); the two cannot coexist under
one `egomimic` import name in one process, which is the entire reason S7 runs in a separate subprocess.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

EGOVERSE_JEPA_PYTHON_EXE = os.environ.get(
    "EGOVERSE_JEPA_PYTHON_EXE", "/mnt/wd10/tongxi/miniconda3/envs/egoverse_jepa/bin/python"
)
S7_WORKER_SCRIPT = os.environ.get(
    "S7_WORKER_SCRIPT",
    "/mnt/wd10/tongxi/egoverse_jepa/egomimic/scripts/experiments/egoverse_a_jepa_n4_2d_6_s7_worker.py",
)


def _egoverse_jepa_subprocess_env() -> dict:
    """Same rationale as SAM3Ops._sam3_subprocess_env: strip the PARENT process's own conda identity
    (this deployment env, e.g. `emimic` or `lerobot`) before launching the worker with a DIFFERENT env's
    interpreter, so the worker's own CUDA/cudnn/torch wheels resolve their own native libraries instead of
    silently picking up the parent env's -- SAM3Ops hit a real cuDNN-version-mismatch crash from exactly
    this class of mistake; not repeating it here."""
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "CONDA_SHLVL",
                "_CONDA_ROOT", "_CONDA_EXE", "LD_LIBRARY_PATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    worker_env_root = os.path.dirname(os.path.dirname(EGOVERSE_JEPA_PYTHON_EXE))
    worker_bin = os.path.join(worker_env_root, "bin")
    other_env_bins = {p for p in os.environ.get("PATH", "").split(os.pathsep)
                       if (os.sep + "envs" + os.sep in p or os.sep + "emimic" + os.sep in p) and p != worker_bin}
    kept_path = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p not in other_env_bins]
    env["PATH"] = os.pathsep.join([worker_bin] + kept_path)
    env["CONDA_PREFIX"] = worker_env_root
    return env


class S7Worker:
    """Long-running cross-env subprocess computing the frozen S7-1-TA spatial-transport token."""

    def __init__(self, device: str = "cuda", timeout_s: float = 30.0):
        if not os.path.exists(EGOVERSE_JEPA_PYTHON_EXE):
            raise RuntimeError(
                f"S7 worker python not found at {EGOVERSE_JEPA_PYTHON_EXE} -- set EGOVERSE_JEPA_PYTHON_EXE."
            )
        if not os.path.exists(S7_WORKER_SCRIPT):
            raise RuntimeError(f"S7 worker script not found at {S7_WORKER_SCRIPT}.")

        env = _egoverse_jepa_subprocess_env()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        if device != "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ""

        print("[S7Worker] Launching persistent worker subprocess (egoverse_jepa env)...")
        self.proc = subprocess.Popen(
            [EGOVERSE_JEPA_PYTHON_EXE, "-u", S7_WORKER_SCRIPT, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )
        self._timeout_s = timeout_s

        init_error = None
        deadline = time.time() + 300  # model loading (resnet+coarse+fine) can take a while cold
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                break
            line = line.strip()
            if line.startswith("{"):
                try:
                    resp = json.loads(line)
                    if not resp.get("ok", True):
                        init_error = resp.get("error", "Unknown init error")
                        break
                except Exception:
                    pass
            if line == "READY":
                print("[S7Worker] Worker subprocess is READY.")
                return

        self.proc.terminate()
        raise RuntimeError(f"S7 worker failed to start: {init_error or 'timed out or exited during init.'}")

    def compute_token(self, img_past: np.ndarray, img_cur: np.ndarray) -> np.ndarray:
        """img_past, img_cur: [3,480,640] float32 arrays in [0,1] (CHW) -- ALREADY resized/cropped by the
        caller to the format the frozen resnet expects (native Aria 480x640, 4:3). Returns a flat [24]
        float32 array (3x4 pooled grid x 2D flow, matching phase_c_s7_precompute.py's own token layout)."""
        tmp_dir = tempfile.gettempdir()
        p_past = os.path.join(tmp_dir, f"s7_past_{uuid.uuid4().hex}.npy")
        p_cur = os.path.join(tmp_dir, f"s7_cur_{uuid.uuid4().hex}.npy")
        try:
            np.save(p_past, img_past.astype(np.float32))
            np.save(p_cur, img_cur.astype(np.float32))
            req = {"cmd": "compute_token", "img_past_path": p_past, "img_cur_path": p_cur}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("S7 worker process died during processing.")
            resp = json.loads(line)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error", "S7 worker error"))
            return np.asarray(resp["token"], dtype=np.float32)
        finally:
            for p in (p_past, p_cur):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def close(self):
        print("[S7Worker] Closing worker subprocess...")
        try:
            self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            if hasattr(self, "proc"):
                self.proc.kill()


_WORKER: S7Worker | None = None


def get_s7_worker(device: str = "cuda") -> S7Worker:
    global _WORKER
    if _WORKER is None:
        _WORKER = S7Worker(device=device)
        atexit.register(_WORKER.close)
    return _WORKER
