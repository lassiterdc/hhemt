"""INVERSE TIER CANARY -- do not "fix", do not delete, do not import.

The mirror of canary_tier.py. `numpy.trapezoid` was ADDED in numpy 2.0 and is
absent in 1.26, so this file is dirty under the CERTIFIED stack and clean under
the REPORTING one -- exactly inverted from its sibling.

    certified stack (numpy 1.26.4)  ->  unresolved-attribute   (expected)
    reporting stack (numpy 2.4.6)   ->  no diagnostic

The pair is a two-arm differential: exactly one probe is dirty and WHICH one names
the tier. A single probe cannot carry that verdict, because a weakened probe is
silent and silence would read as the certified stack. Measured: single-arm gives
2 fail-open rows in 10, the pair gives 0.

Never imported, never executed. If numpy 1.x support is dropped, BOTH probes must
be re-authored together -- re-authoring one inverts the pair's meaning silently.
"""

import numpy

_TIER_DISCRIMINATOR_INVERSE = numpy.trapezoid
