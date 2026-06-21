"""CI guard: a sensitivity-overlay column (system.{field} / analysis.{field}) in
any tracked sensitivity xlsx or test fixture MUST name a field present in the live
analysis_config / system_config model_fields. Catches the field-retirement drift
class (Gotcha 18 / sensitivity_analysis.py:77-94) at commit time, not HPC-run time.
"""

import sys
from pathlib import Path

from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config
from hhemt.sensitivity_analysis import _ANALYSIS_COLUMN_PREFIX, _SYSTEM_COLUMN_PREFIX

REPO = Path(__file__).resolve().parents[1]
_ANALYSIS_FIELDS = set(analysis_config.model_fields)
_SYSTEM_FIELDS = set(system_config.model_fields)


def _overlay_columns(xlsx: Path) -> list[str]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    cols: list[str] = []
    for ws in wb.worksheets:
        header = next(ws.iter_rows(max_row=1, values_only=True), ())
        cols.extend(str(c) for c in header if c and ("." in str(c)))
    wb.close()
    return cols


def _violation(col: str) -> str | None:
    if col.startswith(_SYSTEM_COLUMN_PREFIX):
        field = col[len(_SYSTEM_COLUMN_PREFIX):]
        return None if field in _SYSTEM_FIELDS else f"system.{field}"
    if col.startswith(_ANALYSIS_COLUMN_PREFIX):
        field = col[len(_ANALYSIS_COLUMN_PREFIX):]
        return None if field in _ANALYSIS_FIELDS else f"analysis.{field}"
    return None


def main() -> int:
    violations: list[str] = []
    for xlsx in REPO.rglob("test_data/**/*.xlsx"):
        for col in _overlay_columns(xlsx):
            v = _violation(col)
            if v:
                violations.append(f"{xlsx.relative_to(REPO)}: overlay column '{col}' "
                                  f"names retired field '{v}' (absent from live model_fields)")
    if violations:
        print("Retired-field overlay-column violations:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
