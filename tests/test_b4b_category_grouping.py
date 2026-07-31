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
