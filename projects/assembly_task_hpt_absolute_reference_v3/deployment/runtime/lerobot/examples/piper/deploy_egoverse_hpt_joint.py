#!/usr/bin/env python
"""Compatibility module for Assembly New1 deployment helpers.

The pi0.5 deployment adapters import ``deploy_egoverse_hpt_joint`` for the
audited command queue and hardware gates.  Re-export the independent Assembly
New1 implementation so every profile uses the same three-camera contract.
"""

from deploy_assembly_new1_hpt_joint import *  # noqa: F401,F403

