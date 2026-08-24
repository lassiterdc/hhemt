"""Read provenance labels off an Apptainer container, and compare them to a pin.

LEAF MODULE by design, mirroring ``slurm_liveness.py``: it imports nothing from the
toolkit, so preflight, the setup rung, and any subprocess can all use it without
pulling in ``workflow.py``. The alternative -- a second copy of the read logic at
each call site -- is the drift surface this module exists to remove.
"""

from __future__ import annotations

import json as _json
import shlex as _shlex
import subprocess as _subprocess
from pathlib import Path

TRITON_SHA_LABEL = "org.hhemt.triton_sha"

#: Git's own abbreviation floor. Below this, a prefix match is not evidence of identity.
_MIN_SHA_PREFIX = 7


class ContainerLabelResult:
    """Three-valued outcome of a label read: read / unreadable / read-but-unlabelled.

    The third state is the one that must not collapse into either neighbour. An
    unreadable image is an IMAGE problem; a readable image with no label is a BUILD
    problem; and they send an operator to different remedies.
    """

    def __init__(self, labels=None, read: bool = False, error: str | None = None):
        self.labels = labels or {}
        self.read = read
        self.error = error

    @property
    def triton_sha(self) -> str | None:
        v = self.labels.get(TRITON_SHA_LABEL)
        return str(v) if v else None


def read_container_labels(container_path, apptainer_module=None, timeout=120):
    """Return a ContainerLabelResult for a SIF file or an apptainer SANDBOX directory.

    Measured on UVA Rivanna: sandbox 0.00 s (plain file read, no subprocess), packed
    SIF 0.04-0.71 s. `apptainer` is NOT on the ambient PATH there -- the
    /opt/apptainer/current/bin entry is a dead stub -- so omitting apptainer_module
    makes the read FAIL rather than fall back, which is why the module form is tried
    first and the bare form second.
    """
    p = Path(container_path)
    sandbox_labels = p / ".singularity.d" / "labels.json"
    if sandbox_labels.is_file():
        try:
            return ContainerLabelResult(_json.loads(sandbox_labels.read_text()) or {}, read=True)
        except (OSError, ValueError) as exc:
            return ContainerLabelResult(read=False, error=f"unreadable sandbox labels.json: {exc}")

    inspect = f"apptainer inspect --json {_shlex.quote(str(p))}"
    attempts = ([f"module load {_shlex.quote(apptainer_module)} && {inspect}"] if apptainer_module else []) + [inspect]
    image_error = None
    for cmd in attempts:
        try:
            r = _subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        except (OSError, _subprocess.TimeoutExpired):
            continue
        if r.returncode != 0:
            err = f"{r.stderr}".lower()
            # THREE-VALUED, lifted from setup_workflow.py's measured split. rc 127 or
            # "command not found" -> binary absent; "lmod has detected" -> bad modulefile.
            # Both mean the FORM could not run the tool, so the next form may still answer.
            # Anything else means apptainer RAN and refused the IMAGE, which no other form
            # can change -- stop and keep its own diagnosis.
            if r.returncode == 127 or "command not found" in err or "lmod has detected" in err:
                continue
            image_error = (f"{r.stderr}".strip().splitlines() or ["(no stderr)"])[0]
            break
        try:
            labels = (((_json.loads(r.stdout) or {}).get("data") or {}).get("attributes", {}).get("labels", {})) or {}
        except ValueError as exc:
            return ContainerLabelResult(read=False, error=f"apptainer returned unparseable JSON: {exc}")
        return ContainerLabelResult(labels, read=True)
    return ContainerLabelResult(read=False, error=image_error or "apptainer could not be invoked in any form")


def shas_match(a: str | None, b: str | None) -> bool:
    """True iff two git shas identify the same commit by prefix, at git's own floor.

    Compares at the SHORTER length so a 8-char config pin validates against a 40-char
    label, and refuses anything below _MIN_SHA_PREFIX so a coincidental short prefix
    is never mistaken for identity.
    """
    if not a or not b:
        return False
    a, b = a.strip().lower(), b.strip().lower()
    n = min(len(a), len(b))
    if n < _MIN_SHA_PREFIX:
        return False
    return a[:n] == b[:n]


def looks_like_sha(value: str | None) -> bool:
    """True iff value is a bare hex git object name of at least the abbreviation floor."""
    if not value:
        return False
    v = value.strip()
    return len(v) >= _MIN_SHA_PREFIX and all(c in "0123456789abcdefABCDEF" for c in v)


def sha_in_filename(container_path) -> str | None:
    """Return the longest sha-shaped hex token in the basename, or None.

    Corroboration only, never the authority: a filename is written by whoever last
    moved the file, while the label is written at build time by a recipe the build
    script verifies. Disagreement between them is what identifies a RENAMED or COPIED
    image, which no single-source check can see. Absent on the legacy images, which
    carry no sha in the name at all -- hence None rather than an error.
    """
    stem = Path(container_path).name
    best = None
    for tok in stem.replace(".", "_").replace("-", "_").split("_"):
        if looks_like_sha(tok) and (best is None or len(tok) > len(best)):
            best = tok
    return best
