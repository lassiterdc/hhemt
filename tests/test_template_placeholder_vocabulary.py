"""Every ${...} placeholder in a shipped case-study template is a key of the filler's mapping.

`utils.fill_template` uses `string.Template.safe_substitute`, which leaves an unmatched
placeholder in place SILENTLY. A typo'd `${DAT_DIR}` therefore survives the filler and
reaches the config loader as a literal path, where it now surfaces as a confusing
existence error rather than as the typo it is. This is the static half of the repair;
the load mode is the other half and neither substitutes for the other.
"""

import re
from pathlib import Path

import pytest

from hhemt.experiments import TRITON_SWMM_experiment

CASE = "norfolk_coastal_flooding"
_TEMPLATES = ("template_system_config.yaml", "template_analysis_config.yaml")
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@pytest.mark.parametrize("template_name", _TEMPLATES)
def test_every_template_placeholder_is_a_mapping_key(template_name, tmp_path):
    # DERIVED, never restated: a fourth mapping key cannot make this test stale.
    vocabulary = set(
        TRITON_SWMM_experiment._get_case_data_and_package_directory_mapping_dict(
            case_name=CASE, example_data_dir=tmp_path
        )
    )
    text = Path(TRITON_SWMM_experiment._load_config_filepath(CASE, template_name)).read_text(encoding="utf-8")
    used = set(_PLACEHOLDER.findall(text))
    unknown = sorted(used - vocabulary)
    assert not unknown, (
        f"{template_name} uses placeholder(s) {unknown} that the case mapping does not "
        f"supply. safe_substitute leaves these in place SILENTLY, so they reach the "
        f"config loader as literal paths. Known keys: {sorted(vocabulary)}."
    )
    assert used, f"{template_name} carries no placeholders at all — is it still a template?"
