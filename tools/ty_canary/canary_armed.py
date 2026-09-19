"""CANARY -- NOT named in the [[tool.ty.overrides]] table.

This file's defect MUST be reported. If it is not, the overrides table is
over-broad, ty is not running, or the rule below was renamed upstream.
Do not "fix" this file. See scripts/check_ty_overrides_in_force.py.
"""

canary_value: int = "deliberate invalid-assignment -- see module docstring"
