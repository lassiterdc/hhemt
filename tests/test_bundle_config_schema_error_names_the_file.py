"""A bundle config that fails schema validation is reported against the file it came from.

The problem this file guards: `load_bundle_config` is the one reader of a render bundle's
archived `cfg_analysis.yaml` / `cfg_system.yaml`, reached from `Bundle.from_directory`,
`load_eda_context` and the combine step. pydantic's own error names the model and the field but
never the file, so a user holding a broken bundle is told *what* is wrong and not *where*. The
reader therefore re-raises the schema error as a `ConfigurationError` carrying the file -- and
only the schema error: a missing file or unparseable YAML keeps its own class, because those are
different failures with different remedies. The existing test pins only the exception class and
the field name, so a wrap that dropped the file would pass it.

Nothing here builds an analysis: the reader is called directly on small YAML files.
"""

from pathlib import Path

import pydantic
import pytest
import yaml

from hhemt.bundle._path_policy import load_bundle_config
from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError

_FIXTURE = Path(__file__).parent / "fixtures" / "bundles" / "multi_sim" / "cfg_analysis.yaml"


def _analysis_without_report(path: Path) -> Path:
    """A real bundle analysis config with its required `report` block removed."""
    cfg = yaml.safe_load(_FIXTURE.read_text())
    cfg.pop("report", None)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _empty_mapping(path: Path) -> Path:
    """A config that is a valid YAML mapping and satisfies no required field."""
    path.write_text("{}\n")
    return path


@pytest.mark.parametrize(
    ("writer", "model", "name"),
    [
        pytest.param(_analysis_without_report, analysis_config, "cfg_analysis.yaml", id="analysis-missing-report"),
        pytest.param(_empty_mapping, system_config, "cfg_system.yaml", id="system-empty"),
    ],
)
def test_schema_error_is_a_configuration_error_naming_the_bundle_config_file(tmp_path, writer, model, name):
    """Property: a schema-invalid bundle config raises `ConfigurationError` naming its file.

    Class: every schema violation in either archived config a bundle carries, read through
    `load_bundle_config`. Why it is required: the reader serves three callers, and the file is
    the one fact pydantic's error leaves out; the CLI maps `ConfigurationError` to exit 2 with a
    `Configuration Error` label, while a bare pydantic error does not carry the path at all. What
    kills it: re-raising a pydantic error unwrapped, wrapping it without the file name or path,
    or breaking the chain to the original error. A second correct implementation under which it
    still passes: building the message differently, since only the class, the presence of the
    file's name and path in the rendered error, and the chained cause are asserted.
    """
    cfg_path = writer(tmp_path / name)

    with pytest.raises(ConfigurationError) as excinfo:
        load_bundle_config(cfg_path, model, tmp_path)

    rendered = str(excinfo.value)
    assert name in rendered, rendered
    assert str(cfg_path) in rendered, rendered
    assert isinstance(excinfo.value.__cause__, pydantic.ValidationError), excinfo.value.__cause__


def test_non_schema_failures_keep_their_own_class(tmp_path):
    """Property: only the schema error is wrapped; a missing file and bad YAML keep their classes.

    Class: the two failures `load_bundle_config` meets before validation -- the file is absent,
    and the file is not parseable YAML. Why it is required: those failures have different
    remedies from a schema violation (fetch the file; fix its syntax), and a reader that wrapped
    everything into `ConfigurationError` would tell a user with a truncated download that their
    config was invalid. What kills it: widening the `try` to cover the read or the parse. A
    second correct implementation under which it still passes: a reader that raises a more
    specific subclass of either error, since only `isinstance` is asserted.
    """
    with pytest.raises(FileNotFoundError):
        load_bundle_config(tmp_path / "cfg_analysis.yaml", analysis_config, tmp_path)

    bad = tmp_path / "cfg_system.yaml"
    bad.write_text("key: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_bundle_config(bad, system_config, tmp_path)
