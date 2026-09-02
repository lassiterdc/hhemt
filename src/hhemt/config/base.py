from pathlib import Path
from typing import Any, get_args

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_core import PydanticUndefined
from tabulate import tabulate

from hhemt.plot_utils import print_json_file_tree

# --- Declarative field metadata: ONE declaration, TWO consumers --------------
#
# A field's conditional applicability, its conditional requiredness, and its
# closed-set option glossary are declared HERE -- on the field, inside
# `Field(json_schema_extra=...)`. Two consumers read the SAME declaration:
#
#   (1) `cfgBaseModel._enforce_required_when` below  -- validation
#   (2) `hhemt.report_renderers.metadata`            -- the Reproduction Guide
#
# There is no second, hand-written enforcement site, so the rendered "Required"
# cell CANNOT disagree with what is enforced -- which is the defect this
# replaces: `hpc_total_job_duration_min`'s description named
# `1_job_many_srun_tasks` while its validator required it under `batch_job`.
#
# Precedent: `json_schema_extra={"toolkit_owned_output": True}` is already read
# by three consumers (validation.py, `_check_paths_exist` below, and
# metadata._is_toolkit_owned). This extends that idiom rather than opening a
# second metadata channel.
#
# PINNED-VERSION CONSTRAINT: pydantic is pinned to 2.7.* (pyproject.toml). The
# additive merge of STACKED `json_schema_extra` payloads landed in 2.9, so a
# field must carry exactly ONE `json_schema_extra=`; build the whole payload
# with `field_meta()`.


def when(field: str, *values: Any) -> dict[str, Any]:
    """One trigger clause: `field` holds one of `values`.

    A LIST, not a scalar equality, because five of the conditional fields in
    this codebase are required when a toggle is FALSE -- `when("toggle_x", False)`
    must be expressible in the same grammar as `when("mode", "batch_job")`.
    """
    return {"field": field, "in": list(values)}


def field_meta(
    *,
    applies_when: list[dict[str, Any]] | None = None,
    required_when: list[dict[str, Any]] | None = None,
    options: dict[str, str] | None = None,
    **passthrough: Any,
) -> dict[str, Any]:
    """Build a field's whole `json_schema_extra` payload.

    `applies_when` and `required_when` are DELIBERATELY separate keys, not one
    conflated flag: `hpc_total_job_duration_min` is applicable under BOTH HPC
    execution methods and required under only one of them, and a single key
    cannot say that without lying about one of the two.

    `passthrough` carries any pre-existing key (today: `toolkit_owned_output`)
    so one payload can hold both without stacking.
    """
    extra: dict[str, Any] = dict(passthrough)
    if applies_when:
        extra["applies_when"] = applies_when
    if required_when:
        extra["required_when"] = required_when
    if options:
        extra["options"] = options
    return extra


def declared(field_info: Any, key: str) -> Any:
    """Read one declaration key off a FieldInfo. `None` when undeclared.

    The presence test is what makes the whole mechanism degrade gracefully: a
    field with no declaration yields `None` at every call site, so both
    consumers take their pre-existing branch unchanged.
    """
    extra = getattr(field_info, "json_schema_extra", None)
    return extra.get(key) if isinstance(extra, dict) else None


def render_clauses(clauses: list[dict[str, Any]]) -> str:
    """Human-readable rendering of trigger clauses, for the report table.

    `multi_sim_run_method is batch_job` / `run_mode is one of gpu, hybrid`.
    """
    parts: list[str] = []
    for clause in clauses:
        values = clause["in"]
        joined = str(values[0]) if len(values) == 1 else "one of " + ", ".join(str(v) for v in values)
        parts.append(f"{clause['field']} is {joined}")
    return "; ".join(parts)


class cfgBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _get_field_descriptions(model_cls):
        data = {field_name: field_info.description or "" for field_name, field_info in model_cls.model_fields.items()}
        sr = pd.Series(data)  # type: ignore
        sr.index.name = "attr_name"  # type: ignore
        sr.name = "desc"  # type: ignore
        return sr

    @staticmethod
    def _get_field_optionality(model_cls):
        """
        Returns a Series with field names as index and True/False for optionality
        """
        data = {}
        for name, field in model_cls.model_fields.items():
            is_optional = field.default is not ... or field.allow_none  # type: ignore
            data[name] = is_optional
        sr = pd.Series(data)  # type: ignore
        sr.index.name = "attr_name"  # type: ignore
        sr.name = "optional"  # type: ignore
        return sr

    def cfg_dic_to_df(self):
        s_vals = pd.DataFrame(self, columns=["attr_name", "val"]).set_index("attr_name")["val"]
        s_descs = self._get_field_descriptions(self.__class__)
        df_vars = pd.concat([s_descs, s_vals], axis=1)
        return df_vars

    def print_files_defined_in_yaml(self):
        print_json_file_tree(self.model_dump())

    def display_tabulate_cfg(self, col1_width=25, col2_width=50, col3_width=50):
        data = self.cfg_dic_to_df()

        lst_rows = []
        for idx, row in data.iterrows():
            vals_as_list = [
                str(idx),
                str(row.desc),
                (  # even coerced as strings, True and False cause line splitting to fail so they need to be modified
                    str(row.val).lower() if str(row.val) in ["True", "False"] else str(row.val)
                ),
            ]
            lst_rows.append(vals_as_list)

        print(
            tabulate(
                lst_rows,  # type: ignore
                headers=[str(data.index.name)] + list(data.columns),  # type: ignore
                tablefmt="grid",
                maxcolwidths=[25, 60, 60],
            )
        )

    # VALIDATION
    @staticmethod
    def validate_from_toggle(
        values: dict[str, Any],
        toggle_varname: str,
        lst_rqrd_if_true: list[str],
        lst_rqrd_if_false: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Validate that required fields are provided depending on a toggle.

        Additionally, for fields that are Path-like, validate that the file exists.

        Returns:
            failing_vars: list of field names that failed
            errors: list of error messages
        """
        failing_vars: list[str] = []
        errors: list[str] = []
        toggle = values.get(toggle_varname)
        required_fields = lst_rqrd_if_true if toggle else lst_rqrd_if_false
        for var in required_fields:
            val = values.get(var)
            # Check for presence
            if val is None:
                errors.append(f"{var} must be provided if {toggle_varname} is {'True' if toggle else 'False'}")
                failing_vars.append(var)
                continue
            # Check if Path exists
            if isinstance(val, Path):
                p = val.expanduser()
                if not p.exists():
                    errors.append(f"{var} path does not exist: {p}")
                    failing_vars.append(var)
        return failing_vars, errors

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Fail at IMPORT time when an `options` glossary drifts from its Literal.

        This is the structural anti-drift guarantee for closed-set fields. A test
        can be skipped or forgotten; an import cannot. Adding a member to a
        `Literal` without adding it to the glossary raises here, at class-definition
        time, before any test runs.

        Union annotations -- a Literal ORed with a list of that Literal, as on
        `clear_raw` and `reclaim_after_processing` -- are REFUSED rather than
        silently accepted: `get_args` on a union returns the union members, not
        the string values, so the parity check could not actually check anything.
        Fail closed and name the field.
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, field_info in cls.model_fields.items():
            options = declared(field_info, "options")
            if options is None:
                continue
            legal = set(get_args(field_info.annotation))
            if not legal or not all(isinstance(v, str) for v in legal):
                raise TypeError(
                    f"{cls.__name__}.{name}: options= is declared on a field whose "
                    f"annotation is not a flat Literal of strings "
                    f"({field_info.annotation!r}); the option/annotation parity check "
                    f"cannot be performed, so the declaration is refused."
                )
            if set(options) != legal:
                raise TypeError(
                    f"{cls.__name__}.{name}: options keys {sorted(options)} != "
                    f"Literal members {sorted(legal)}. The glossary has drifted from "
                    f"the type; update whichever is stale."
                )

    @model_validator(mode="before")
    @classmethod
    def _enforce_required_when(cls, values: Any) -> Any:
        """Enforce every `required_when` declaration on this model. THE ONLY SITE.

        Generic over all fields, so a new conditional requirement is landed by
        annotating the field and nothing else. There is deliberately no
        per-field hand-written counterpart to fall out of step with the
        rendered table.

        `mode="before"` sees RAW input, so a trigger field absent from `values`
        must fall back to its DECLARED DEFAULT. A naive `values.get(trigger_name)`
        returns None and silently exempts every YAML that omits the trigger --
        and `multi_sim_run_method` defaults to "local", so omission is the
        common case and the bug would be invisible.
        """
        if not isinstance(values, dict):
            return values
        errors: list[str] = []
        for name, field_info in cls.model_fields.items():
            clauses = declared(field_info, "required_when")
            if not clauses:
                continue
            for clause in clauses:
                trigger_name = clause["field"]
                if trigger_name in values:
                    trigger = values[trigger_name]
                else:
                    trigger_field = cls.model_fields.get(trigger_name)
                    if trigger_field is None:
                        raise TypeError(
                            f"{cls.__name__}.{name}: required_when names an unknown " f"field {trigger_name!r}"
                        )
                    default = trigger_field.default
                    trigger = None if default is PydanticUndefined else default
                if trigger in clause["in"] and values.get(name) is None:
                    errors.append(f"{name} is required when {render_clauses([clause])}")
        if errors:
            raise ValueError("; ".join(errors))
        return values

    @field_validator("*", mode="before")
    @classmethod
    def _check_paths_exist(cls, v: Any, info) -> Any:
        """
        Validate that all Path-like fields exist.
        Skips non-path fields automatically, and skips fields marked
        json_schema_extra={"toolkit_owned_output": True} — those name
        directories the clone/build gate CREATES at run/setup time, so
        they legitimately do not exist at config-load time.
        """
        if v is None:
            return v  # allow optional
        # Skip toolkit-owned OUTPUT path fields (created by the clone/build
        # gate at run/setup, not user-provided inputs). Must precede the
        # isinstance(v, Path) check because model_dump() (mode="python")
        # passes these fields as Path objects, so the existence branch would
        # otherwise fire. See system.py TRITONSWMM/SWMM clone gates.
        field_info = cls.model_fields.get(info.field_name)
        extra = field_info.json_schema_extra if field_info is not None else None
        if isinstance(extra, dict) and extra.get("toolkit_owned_output"):
            return v
        # Only handle Path or str values
        if isinstance(v, Path):
            p = Path(v).expanduser()
            if not p.exists():
                raise ValueError(f"File does not exist: {p}")
            return p  # convert str → Path
        # everything else is ignored
        return v
