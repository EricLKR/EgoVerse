#!/usr/bin/env python
"""Module alias for the Assembly New1 deployment helpers.

The pi0.5 deployment adapters import ``deploy_egoverse_hpt_joint`` for the
audited command queue and hardware gates.  The adapters temporarily replace
``CHECKPOINT_PROFILES`` while parsing their own profile.  A wildcard re-export
would copy that dictionary and leave the real parser's globals unchanged, so
this compatibility name must resolve to the implementation module itself.
"""

import sys

import deploy_assembly_new1_hpt_joint as _implementation

sys.modules[__name__] = _implementation
