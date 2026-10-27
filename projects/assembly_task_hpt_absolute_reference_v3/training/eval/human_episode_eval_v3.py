#!/usr/bin/env python3
"""Enable the audited human episode evaluator for Absolute-Reference V3."""

from __future__ import annotations

import os
import sys
from pathlib import Path


script_dir = Path(os.environ["ASSEMBLY_NEW1_EVAL_SCRIPT_DIR"])
sys.path.insert(0, str(script_dir))

import human_episode_eval as evaluator  # noqa: E402


evaluator.SUPPORTED_HEAD_CLASSES.add("AbsoluteReferenceJointPolicyV3")


if __name__ == "__main__":
    raise SystemExit(evaluator.main())
