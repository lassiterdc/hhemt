"""TIER CANARY -- do not "fix", do not delete, do not import.

Asserts which dependency stack `ty` actually resolved, by observing a numpy-major
discriminator: `numpy.trapz` exists in numpy 1.x and was REMOVED in numpy 2.x.

    certified stack (numpy 1.26.4, per environment.yaml)  ->  no diagnostic
    reporting stack (numpy 2.4.6, the pip/.venv tier)     ->  unresolved-attribute

Ruling D144 gates on the certified stack and makes the reporting stack non-blocking,
so scripts/check_ty_overrides_in_force.py --gating requires this file to be CLEAN.

This file is never imported and never executed; it exists to be type-checked. If
numpy 1.x support is ever dropped the discriminator inverts and this file must be
re-authored -- which is the same condition D144 keys its own flip on.
"""

import numpy

_TIER_DISCRIMINATOR = numpy.trapz
