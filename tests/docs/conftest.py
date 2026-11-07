"""Mark every test under `tests/docs/` with the `docs` marker at collection.

The documentation tests compare a published page with the code it describes.
Selecting them with `-m docs` must find every one of them, including a test
added later by an author who forgets the marker. A per-file `pytestmark`
depends on that author remembering it; a directory hook marks the test regardless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DOCS_TEST_DIR = Path(__file__).resolve().parent


# `tryfirst=True`, as in tests/conftest.py, so this hook sets the marker before the
# collection hooks that carry no `tryfirst`.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Add `pytest.mark.docs` to every collected item whose file lies under `tests/docs/`.

    pytest calls this hook with the items of the whole session, not of this
    directory alone, so the path test is what keeps the marker off every other test.
    """
    for item in items:
        if _DOCS_TEST_DIR in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.docs)
