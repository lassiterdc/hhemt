"""The EDA configuration states no unbuilt work and carries the setting it promised.

An EDA kind the render refuses is refused at render, and ``dem_resolution_tolerance_m`` exists
with a default of ``None``. No test here
compiles or runs a solver or reaches a ``*_compiled`` or ``synthetic_*_completed`` fixture.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import hhemt.config.eda as eda_module
from hhemt.config.eda import _EDA_DROPS, _EDA_RENAMES, _RETIRED_EDA_FIGURE_STEMS, eda_config
from hhemt.eda._plotting import render_eda_plots


def test_a_retired_eda_kind_is_refused_at_render_and_a_current_kind_is_not(tmp_path):
    """Every kind the render refuses is refused by the render-time kind validation, as an unknown kind is.

    Property: ``render_eda_plots`` raises ``ValueError`` naming an unknown EDA plot kind for
    each key of ``_RETIRED_EDA_FIGURE_STEMS``, and the config load that carries it neither
    warns nor rewrites the list.

    Class: every figure stem the render refuses, moved to another stem or dropped, in a saved or baked
    ``cfg_analysis.yaml``. The set is read from the module, so a stem added to it later is
    covered without a new test.

    Why it is required: a rewrite that turned such a kind into its successor and warned that the
    key "will be rejected in a future release" would promise a refusal it did not make. The
    refusal is the behavior that promise describes.

    What kills it: restoring a before-validator that rewrites or drops the key (the load
    would warn or change the list), or registering a refused stem in ``_EDA_RENDERERS`` (the
    render would not raise).

    A second correct implementation under which it still passes: refusing such a kind at
    config load with a different exception type would fail the render arm, so this test also
    pins the render-time gate, which is the documented enforcement point.

    Differently positioned satisfying input: a current kind (``raw_byte_identity``) renders
    its honest-degradation panel with no backing artifact, so the refusal is selective.
    """
    assert _RETIRED_EDA_FIGURE_STEMS, "the set of refused stems is empty, so the loop below would test nothing"
    assert _RETIRED_EDA_FIGURE_STEMS == frozenset(_EDA_RENAMES) | _EDA_DROPS
    for stem in sorted(_RETIRED_EDA_FIGURE_STEMS):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning at load is a failure
            cfg = eda_config(enabled_plots=["config_diff_maps", stem])
        assert cfg.enabled_plots == ["config_diff_maps", stem], f"load rewrote {stem}"
        (tmp_path / "eda").mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="unknown EDA plot kind") as caught:
            render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=cfg)
        assert stem in str(caught.value)

    paths = render_eda_plots(
        tmp_path, cfg_analysis=None, eda_cfg=eda_config(enabled_plots=["raw_byte_identity"], plotly_js_mode="cdn")
    )
    assert [p.name for p in paths] == ["raw_byte_identity.html"]


def test_the_module_promises_no_future_release():
    """No field description and no source line of ``config/eda.py`` promises a future release.

    Property: the phrase ``future release`` appears in no field description of ``eda_config``
    and nowhere in the module source.

    Class: the wording of a warning for a refused kind, which the planning-cruft rule names for this
    file.

    What kills it: re-adding the deprecation warning text to a description or a comment.

    A second correct implementation under which it still passes: rewording any description to
    other text without the phrase.
    """
    source = inspect.getsource(eda_module).lower()
    descriptions = " ".join((f.description or "") for f in eda_config.model_fields.values()).lower()
    assert "future release" not in source
    assert "future release" not in descriptions


def test_dem_resolution_tolerance_defaults_to_none_and_accepts_only_positive_meters():
    """``dem_resolution_tolerance_m`` is absent by default and takes a positive value.

    Property: the default is ``None``; ``0.05`` loads; ``0`` and ``-0.05`` are refused,
    because a log-scaled error axis cannot draw a line at a non-positive depth error.

    Class: the field's value space (unset, positive, non-positive).

    What kills it: a non-``None`` default (a tolerance nobody declared), removing the
    field (``extra="forbid"`` refuses the key), or dropping the positivity bound.

    A second correct implementation under which it still passes: an ``Annotated`` positive
    float type in place of ``Field(gt=0)``; the assertions read only behavior.
    """
    assert eda_config().dem_resolution_tolerance_m is None
    assert eda_config(dem_resolution_tolerance_m=0.05).dem_resolution_tolerance_m == 0.05
    for bad in (0, -0.05):
        with pytest.raises(Exception, match="dem_resolution_tolerance_m"):
            eda_config(dem_resolution_tolerance_m=bad)
