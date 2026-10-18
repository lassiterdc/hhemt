"""Q4a (iter-2): the b4b figure groups DISPLAY panels by hardware CATEGORY (one CPU panel,
one GPU panel) instead of per-GPU-hardware faceting, while each row keeps its OWN family's
within-family reference math. These are the pure category-collapse helpers."""

from hhemt.eda._plotting import _b4b_category, _b4b_category_title


def test_b4b_category_collapses_gpu_hardware_to_one_gpu_category():
    # cpu stays cpu; ANY GPU hardware token collapses to 'gpu'; the degraded sentinel stays 'all'.
    assert _b4b_category("cpu") == "cpu"
    assert _b4b_category("a6000") == "gpu"
    assert _b4b_category("a100-80") == "gpu"
    assert _b4b_category("all") == "all"


def test_b4b_category_grouping_yields_one_cpu_one_gpu_panel():
    # the per-hardware families {cpu, a6000, a100-80} collapse to exactly {cpu, gpu} categories.
    families = ["cpu", "a6000", "a100-80"]
    categories = sorted({_b4b_category(f) for f in families})
    assert categories == ["cpu", "gpu"]


def test_b4b_category_title_is_human():
    assert _b4b_category_title("cpu") == "CPU"
    assert _b4b_category_title("gpu") == "GPU"
    assert _b4b_category_title("all") == "All configs"


def test_absent_artifact_panel_names_a_bundle_emitted_before_the_rename(tmp_path):
    """The raw byte-identity panel says a pre-rename bundle predates the rename.

    Property: when ``eda/raw_byte_identity.zarr`` is absent but the zarr of its predecessor
    in ``config/eda.py``'s ``_EDA_RENAMES`` is present, the degradation panel says the bundle
    predates the rename and should be re-emitted, and does NOT say the calc member did not
    run. With neither present, the original reason stands.

    Class: every render root carrying the predecessor's backing artifact and not the new
    one, which is the state of a render bundle emitted before the rename.

    Why it is required: the original reason is false there, because the calc member did run
    and its data sits beside the panel under the old name.

    What kills it: keying the branch on ``_RETIRED_EDA_FIGURE_STEMS`` (the control below
    carries ``eda_cross_sim_identity.zarr``, a retired stem whose zarr is a live backing
    artifact, and must keep the original reason), or dropping the branch.

    A second correct implementation under which it still passes: resolving the predecessor
    once at import time instead of per render; the assertions read only the panel text.
    """
    from hhemt.config.eda import _EDA_RENAMES, eda_config
    from hhemt.eda._plotting import render_eda_plots

    predecessors = [old for old, new in _EDA_RENAMES.items() if new == "raw_byte_identity"]
    assert len(predecessors) == 1, predecessors

    def panel_text(root):
        [out] = render_eda_plots(root, cfg_analysis=None, eda_cfg=eda_config(enabled_plots=["raw_byte_identity"]))
        text = out.read_text()
        for line_break in ("<br>", "\\u003cbr\\u003e"):
            text = text.replace(line_break, " ")
        return text

    bundle = tmp_path / "pre_rename_bundle"
    (bundle / "eda" / f"{predecessors[0]}.zarr").mkdir(parents=True)
    text = panel_text(bundle)
    assert "predates" in text
    assert "calc member did not run" not in text

    control = tmp_path / "control"
    (control / "eda" / "eda_cross_sim_identity.zarr").mkdir(parents=True)
    text = panel_text(control)
    assert "predates" not in text
    assert "calc member did not run" in text
