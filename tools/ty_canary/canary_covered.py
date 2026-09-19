"""CANARY -- IS named in the [[tool.ty.overrides]] table.

This file carries the IDENTICAL defect to canary_armed.py and MUST NOT be
reported. If it is, the overrides table is not being honoured.
Do not "fix" this file. See scripts/check_ty_overrides_in_force.py.
"""

canary_value: int = "deliberate invalid-assignment -- see module docstring"
