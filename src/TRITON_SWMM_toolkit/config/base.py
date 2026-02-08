from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree


class cfgBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _get_field_descriptions(model_cls):
        data = {
            field_name: field_info.description or ""
            for field_name, field_info in model_cls.model_fields.items()
        }
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
        s_vals = pd.DataFrame(self, columns=["attr_name", "val"]).set_index(
            "attr_name"
        )["val"]
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
                    str(row.val).lower()
                    if str(row.val) in ["True", "False"]
                    else str(row.val)
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
        values: Dict[str, Any],
        toggle_varname: str,
        lst_rqrd_if_true: List[str],
        lst_rqrd_if_false: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that required fields are provided depending on a toggle.

        Additionally, for fields that are Path-like, validate that the file exists.

        Returns:
            failing_vars: list of field names that failed
            errors: list of error messages
        """
        failing_vars: List[str] = []
        errors: List[str] = []
        toggle = values.get(toggle_varname)
        required_fields = lst_rqrd_if_true if toggle else lst_rqrd_if_false
        for var in required_fields:
            val = values.get(var)
            # Check for presence
            if val is None:
                errors.append(
                    f"{var} must be provided if {toggle_varname} is {'True' if toggle else 'False'}"
                )
                failing_vars.append(var)
                continue
            # Check if Path exists
            if isinstance(val, Path):
                p = val.expanduser()
                if not p.exists():
                    errors.append(f"{var} path does not exist: {p}")
                    failing_vars.append(var)
        return failing_vars, errors

    @field_validator("*", mode="before")
    @classmethod
    def _check_paths_exist(cls, v: Any, info) -> Any:
        """
        Validate that all Path-like fields exist.
        Skips non-path fields automatically.
        """
        if v is None:
            return v  # allow optional
        # Only handle Path or str values
        if isinstance(v, Path):
            p = Path(v).expanduser()
            if not p.exists():
                raise ValueError(f"File does not exist: {p}")
            return p  # convert str → Path
        # everything else is ignored
        return v
