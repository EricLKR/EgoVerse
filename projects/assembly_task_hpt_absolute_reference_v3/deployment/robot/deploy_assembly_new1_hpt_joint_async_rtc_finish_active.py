#!/usr/bin/env python
"""Assembly New1 HPT with asynchronous inference and committed chunk tails.

This additive entrypoint keeps the audited asynchronous RTC implementation but
defaults to two opt-in scheduler changes:

* returned arm chunks are buffered until the active 30-action chunk completes;
* confirmed gripper object contact requests one fresh observation immediately.

The new arm result never interrupts the active chunk.  At the boundary it is
activated at the absolute-time-aligned offset, while the existing driver-space
step and acceleration limiter remains the continuity authority.
"""

from __future__ import annotations

import sys

import deploy_assembly_new1_hpt_joint_async_rtc as base


def _add_boolean_default(enabled: str, disabled: str) -> None:
    if enabled not in sys.argv and disabled not in sys.argv:
        sys.argv.append(enabled)


def main() -> None:
    _add_boolean_default(
        "--rtc-finish-active-plan-before-install",
        "--no-rtc-finish-active-plan-before-install",
    )
    _add_boolean_default(
        "--rtc-replan-after-contact",
        "--no-rtc-replan-after-contact",
    )
    base.main()


if __name__ == "__main__":
    main()
