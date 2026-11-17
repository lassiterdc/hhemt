"""The Norfolk example config and the container-form coupling judge, against a Norfolk-shaped model.

The Norfolk system config template names one runnable model and a pinned TRITON commit. It lists no outfall to
couple: the analysis config template names the two coupled outfalls, ``E143250`` and ``E147007``, as the outfalls
of the water-level record of the external-boundary layer, and the toolkit converts them. The judge
``require_triton_couples_the_listed_nodes`` decides, from the commit alone in container form, whether a run
couples the toolkit's list. The published model (``hague_model.inp``, about 29,000 lines) is not in the
repository, so the judge tests run on a small stand-in that has the properties the judge reads: two outfalls at
conduit ends that the config converts, one outfall it does not, and no ``[INFLOWS]`` section. The stand-in is
derived with the outfalls the analysis template maps to a water-level variable, so a change to that map in the
template reaches the judge.

Laptop tier. No solver, no network, no clone of TRITON.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr
import yaml

from hhemt.exceptions import ConfigurationError
from hhemt.scenario import InpText, require_triton_couples_the_listed_nodes
from hhemt.swmm_runnable import derive_role_model
from hhemt.swmm_utils import swmm_model_problems
from hhemt.water_level_boundary import converted_node_depths

_CASE_DIR = Path(__file__).resolve().parents[1] / "test_data" / "norfolk_coastal_flooding"

# The head of the fork's ``develop`` branch, whose ``src/config_utils.h`` reads ``swmm_coupling_nodes``.
PIN = "deefa6e6fcb26184a60eb8242dc82dcc3cab9618"
# A commit whose ``[INFLOWS]`` read the toolkit recognizes and which does not read ``swmm_coupling_nodes``:
# the commit the published coupled Norfolk run used.
KEYLESS_PIN = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"

_RETIRED_TEMPLATE_FILES = (
    "TRITON_SWMM_definition_template.cfg",
    "hague_full_model_template.inp",
    "hague_hydraulics_model_template.inp",
    "hague_hydrology_model_template.inp",
)


# A four-cell DEM of 1 m cells whose ground is one constant, built once at collection so the first CRS construction
# of the process is charged to no test. Cell centres sit at x = 0.5 ... 3.5 and y = 1.5, 0.5.
def _flat_dem(ground: float) -> xr.DataArray:
    cells = xr.DataArray(
        np.full((2, 4), ground, dtype="float32"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]},
        dims=("y", "x"),
    )
    return cells.rio.write_crs("EPSG:32618")


_GROUND_ABOVE_INVERT = _flat_dem(0.5)
_GROUND_BELOW_INVERT = _flat_dem(-2.0)

_JUDGE = dict(one_per_cell=False, execution_environment="container", software_directory=None)

# Sections the derivation and the judge read. The two listed outfalls each end a conduit, as the Norfolk outfalls do.
_STAND_IN = """[TITLE]
Norfolk-shaped stand-in

[OPTIONS]
FLOW_UNITS           CMS
FLOW_ROUTING         DYNWAVE

[JUNCTIONS]
;;Name  Elev  MaxDepth  Init  Sur  Aponded
J1      1.0   3.0       0     0    0

[OUTFALLS]
;;Name  Elev   Type  Gated
E143250 -1.079 FREE  NO
E147007 -1.079 FREE  NO
OTHER   0.5    FREE  NO

[CONDUITS]
;;Name Node1 Node2   Length Rough InOff OutOff InitFlow MaxFlow
C1     J1    E143250 10     0.013 0     0      0        0
C2     J1    E147007 10     0.013 0     0      0        0
C3     J1    OTHER   10     0.013 0     0      0        0

[XSECTIONS]
;;Link Shape    G1 G2 G3 G4 Barrels
C1     CIRCULAR 1  0  0  0  1
C2     CIRCULAR 1  0  0  0  1
C3     CIRCULAR 1  0  0  0  1

[COORDINATES]
J1      0 0
E143250 1 0
E147007 2 0
OTHER   3 0
"""

# The same model with an ``[INFLOWS]`` block that names exactly the two listed outfalls, one line each. A build
# that does not read ``swmm_coupling_nodes`` couples the nodes of this block.
_INFLOWS_NAMING_THE_LISTED = """
[INFLOWS]
;;Node    Constituent Time Series
;;------- ----------- -----------
E143250   FLOW        q
E147007   FLOW        q

"""


def _system_template() -> dict:
    return yaml.safe_load((_CASE_DIR / "template_system_config.yaml").read_text())


def _analysis_template() -> dict:
    return yaml.safe_load((_CASE_DIR / "template_analysis_config.yaml").read_text())


def _listed() -> list[str]:
    """The outfalls the toolkit converts for Norfolk: the ones the analysis template maps to a water-level variable."""
    return sorted(_analysis_template()["water_level_boundary_outfall_variables"])


def _rows(text: str, section: str) -> list[str]:
    """First token of each data row of ``[section]`` in SWMM text (comment and blank lines skipped)."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("["):
            inside = line.strip().upper() == f"[{section}]"
        elif inside and line.strip() and not line.lstrip().startswith(";"):
            out.append(line.split()[0])
    return out


def _hydraulics(text: str = _STAND_IN) -> InpText:
    return InpText(derive_role_model(text, "hydraulics", _listed()))


def test_template_names_the_runnable_model_the_outfall_decisions_and_the_pinned_build():
    """Property: the Norfolk templates set the runnable-model fields, state each outfall decision, no removed key.

    Class: REGRESSION, red on the tree before this chunk (the template named three template files and a cfg
    template, and pinned ``15eb18a5``). The input class is every key of the one tracked template.
    Reachability: the template is the file ``hhemt`` copies for a new Norfolk analysis, so a removed key left in
    it makes the toolkit refuse the config at load.
    Both-states anchor: the assertions read the template's keys, not wording the change introduced.
    Why required: the removed keys are refused at load, so a template that keeps one cannot start a run.
    What kills it: restoring ``SWMM_hydraulics`` or the old pin in the template.
    A second correct implementation: a template that sets ``SWMM_model`` to another path under ``${DATA_DIR}/external``.
    """
    cfg = _system_template()
    assert cfg["SWMM_model"].startswith("${DATA_DIR}/external/") and cfg["SWMM_model"].endswith(".inp")
    # Every outfall decision is stated once: the system template converts none directly, and the analysis template
    # names the two Norfolk outfalls as the outfalls of the layer's water-level record, keeps none, and gives their
    # variable its unit and offset, the layer columns and the DEM that sets the converted nodes' depth.
    assert cfg["swmm_outfalls_to_couple"] == []
    analysis = _analysis_template()
    assert _listed() == ["E143250", "E147007"]
    assert analysis["swmm_outfalls_to_keep"] == []
    assert analysis["external_boundary_type_attribute"] == "type"
    assert analysis["water_level_boundary_outfall_attribute"] == "outfalls"
    assert analysis["water_level_boundary_series_source"] == "weather_time_series"
    assert analysis["dem_rim_resolution"] == "processed"
    variables = set(analysis["water_level_boundary_outfall_variables"].values())
    assert set(analysis["water_level_boundary_variable_units"]) == variables == {"waterlevel_m"}
    assert set(analysis["water_level_boundary_variable_datum_offsets"]) == variables
    assert cfg["TRITONSWMM_branch_key"] == PIN
    assert cfg["TRITONSWMM_git_URL"] == "https://github.com/lassiterdc/triton.git"
    removed = {"SWMM_hydraulics", "SWMM_hydrology", "SWMM_full", "triton_swmm_configuration_template"}
    assert removed.isdisjoint(cfg)
    # Control: the matcher fires on a config that still carries a removed key.
    assert not removed.isdisjoint({**cfg, "SWMM_full": "x"})


def test_manifest_lists_the_runnable_model_and_none_of_the_retired_template_files():
    """Property: ``case.yaml`` pins the runnable model by a sha256 digest and lists no removed template file.

    Class: REGRESSION, red on the tree before this chunk (the manifest held the four removed files).
    Reachability: the fetch verifies exactly the manifest keys, so a removed key makes the fetch ask the store
    for a file the new resource version no longer holds.
    Both-states anchor: the test reads manifest keys and digests, not wording.
    What kills it: leaving any of the four removed rows in the manifest, or dropping the model's row.
    A second correct implementation: a manifest that pins the model under any digest, since the digest changes
    on every re-derivation of the published file.
    """
    manifest = yaml.safe_load((_CASE_DIR / "case.yaml").read_text())["manifest"]
    model_key = "data/contents/" + _system_template()["SWMM_model"].removeprefix("${DATA_DIR}/")
    assert re.fullmatch(r"[0-9a-f]{64}", manifest[model_key])
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["data/contents/external/variable_bc.gpkg"])
    assert not {f"data/contents/external/{name}" for name in _RETIRED_TEMPLATE_FILES} & set(manifest)
    # Control: the matcher fires on a manifest that still lists a removed file.
    assert {f"data/contents/external/{name}" for name in _RETIRED_TEMPLATE_FILES} & {
        "data/contents/external/hague_full_model_template.inp"
    }


def test_listed_outfalls_become_junctions_and_the_pinned_build_passes_the_container_judge():
    """Property: with the template list and pin, the listed outfalls are junctions and the container judge passes.

    Class: NEW CAPABILITY. The plausible wrong implementation is a container set that lacks the pinned commit,
    which the judge then refuses as listed in neither set of known commits; the test is red under it.
    Reachability: a container run records the image commit, which equals the template's pin, so the judge
    looks the pin up in the set of commits that read ``swmm_coupling_nodes``.
    Both-states anchor: the assertion turns on the judge returning, which holds for any set that admits the pin.
    Why required: a refusal here stops every Norfolk container run before the first job.
    What kills it: deleting the pin from ``SWMM_COUPLING_NODES_CONTAINER_SHAS``.
    A second correct implementation: a judge that reads ``config_utils.h`` of the image instead of a set.
    """
    hydraulics = _hydraulics()
    junctions, outfalls = _rows(hydraulics, "JUNCTIONS"), _rows(hydraulics, "OUTFALLS")
    assert {"E143250", "E147007"} <= set(junctions)
    assert not {"E143250", "E147007"} & set(outfalls)
    assert "OTHER" in outfalls and not any(n.startswith("hhemt_dummy_") for n in outfalls + junctions)
    assert (
        require_triton_couples_the_listed_nodes(_listed(), hydraulics, branch_key=PIN, recorded_sha=PIN, **_JUDGE)
        is None
    )


def test_a_container_commit_that_differs_from_the_pin_is_refused():
    """Property: the container judge refuses an image whose commit differs from the configured pin.

    Class: NEW CAPABILITY. The plausible wrong implementation is a judge that accepts any recorded commit, so
    the pass in the neighbouring test would carry no information; the test is red under it.
    Reachability: ``recorded_sha`` comes from the image label and ``branch_key`` from the config, and the two
    differ whenever the image was built before a re-pin.
    Both-states anchor: the refusal turns on the inequality of two commits.
    What kills it: dropping the equality test from ``triton_coupling_read``.
    A second correct implementation: a judge that resolves both commits to trees and compares the trees.
    """
    with pytest.raises(ConfigurationError) as caught:
        require_triton_couples_the_listed_nodes(
            _listed(), _hydraulics(), branch_key=PIN, recorded_sha=KEYLESS_PIN, **_JUDGE
        )
    assert caught.value.field == "TRITONSWMM_branch_key"


def test_the_commit_of_the_published_run_is_refused_for_the_converted_outfalls():
    """Property: a build that ignores the coupling list is refused when its ``[INFLOWS]`` read differs from the list.

    Class: NEW CAPABILITY. The plausible wrong implementation is a judge that passes every commit it
    recognizes, so the published ``15eb18a5`` run could be repeated with the converted outfalls left uncoupled;
    the test is red under it.
    Reachability: the stand-in has no ``[INFLOWS]`` section, like the Norfolk model, so that build couples no
    node while the toolkit lists two.
    Both-states anchor: the verdict turns on the two node multisets, and the second arm passes once an
    ``[INFLOWS]`` block names exactly the listed nodes.
    Why required: the converted outfalls would otherwise run uncoupled with no message.
    What kills it: returning ``None`` for the ``inflows`` read without comparing the multisets.
    A second correct implementation: a judge that compares sorted lists after SWMM's case fold.
    """
    with pytest.raises(ConfigurationError) as caught:
        require_triton_couples_the_listed_nodes(
            _listed(), _hydraulics(), branch_key=KEYLESS_PIN, recorded_sha=KEYLESS_PIN, **_JUDGE
        )
    assert caught.value.field == "TRITONSWMM_branch_key"
    assert "E143250" in str(caught.value) and "E147007" in str(caught.value)
    # A differently positioned input: the same commit passes when its read names exactly the listed nodes.
    agreeing = _hydraulics(_STAND_IN + _INFLOWS_NAMING_THE_LISTED)
    assert (
        require_triton_couples_the_listed_nodes(
            _listed(), agreeing, branch_key=KEYLESS_PIN, recorded_sha=KEYLESS_PIN, **_JUDGE
        )
        is None
    )


def test_a_converted_outfall_with_no_conduit_is_accepted_when_its_dem_cell_is_above_its_invert_and_refused_when_not(
    tmp_path,
):
    """Property: an outfall with no conduit is a valid converted outfall when its DEM cell lies above its invert.

    Class: REGRESSION, red on the tree before this chunk: the conduit-end rule (V7) refused a listed outfall with
    no conduit, because only a conduit end crown-raised a node recorded at depth 0. The toolkit now writes a
    converted node's depth from the DEM (DR-35), so the rule is gone and the DEM cell decides.
    Reachability: ``swmm_model_problems`` runs at preflight over the listed outfalls, and ``converted_node_depths``
    reads the cell of each converted node from the DEM that ``dem_rim_resolution`` names.
    Both-states anchor: the model check varies the conduit rows of one model and nothing else; the depth check varies
    the ground of one DEM and nothing else. Neither turns on message wording, apart from the refused node's name.
    What kills it: restoring the conduit-end refusal in ``swmm_model_problems``, or letting a ground at or below the
    invert give a depth.
    A second correct implementation: a check that reads the cell through the processed DEM instead of an in-memory one.
    """
    model = tmp_path / "model.inp"
    cfg = SimpleNamespace(
        SWMM_model=model,
        swmm_outfalls_to_couple=["E147007"],
        subcatchment_raingage_mapping=None,
        subcatchment_raingage_mapping_gage_id_colname=None,
    )
    model.write_text(_STAND_IN)
    assert swmm_model_problems(cfg) == []
    no_conduit = _STAND_IN.replace("C2     J1    E147007 10     0.013 0     0      0        0\n", "")
    assert no_conduit != _STAND_IN, "control: the replacement removed the conduit"
    model.write_text(no_conduit)
    assert swmm_model_problems(cfg) == []
    node = {"E147007": (2.0, 0.0, -1.079)}
    depths, problems = converted_node_depths(_GROUND_ABOVE_INVERT, node)
    assert problems == [] and depths["E147007"] == pytest.approx(0.5 + 1.079)
    depths, problems = converted_node_depths(_GROUND_BELOW_INVERT, node)
    assert depths == {} and [p.code for p in problems] == ["V23"] and "E147007" in problems[0].message
