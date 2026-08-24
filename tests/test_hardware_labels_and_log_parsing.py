"""Device-identity observation: the label map's fallback and the log parser's
absent-field contract.

Both surfaces exist to make simulation hardware an OBSERVED property. Each has one
path that no run available to this campaign can exercise, which is why they are
tested rather than only measured:

- `hardware_label`'s raw-name passthrough is the path a real MI250X takes, and no
  AMD GPU is reachable from the cluster this campaign runs on.
- `parse_triton_log_file`'s absent-`GPU` path is the path EVERY log produced before
  the TRITON emission takes, which is every log this campaign has so far produced.
"""

from __future__ import annotations

import pytest

from hhemt.utils import parse_triton_log_file

#: A real RUN INFO block, copied from a log this repo produced. Note that it carries
#: `GPUs per task :` and `GPU backend :` but NO `GPU :` line -- it predates the
#: emission, which is exactly the state of every pre-existing campaign log.
#: (`#:` rather than `# ` deliberately: this file is quoted verbatim inside a
#: markdown deliverable, and a column-0 `# ` reads as an H1 to a line-based
#: heading scan, truncating the surrounding document at this line.)
_LOG_WITHOUT_GPU_LINE = """TRITON
Machine : laptop
CPU : AMD Ryzen 5 4600H with Radeon Graphics
nTasks : 1
OMP threads per task : 1
GPUs per task : 0 (CPU-only)
GPU backend : none
Total GPUs : 0
Build type : CPU+OMP
TRITON total wall time [s] : 12.5
"""

_LOG_WITH_GPU_LINE = """TRITON
Machine : udc-aw33-4c0
CPU : AMD EPYC 7742 64-Core Processor
GPU : NVIDIA RTX A6000
nTasks : 4
OMP threads per task : 1
GPUs per task : 1
GPU backend : CUDA
Total GPUs : 4
Build type : GPU+CUDA
TRITON total wall time [s] : 265.2
"""


def _write_log(tmp_path, text):
    p = tmp_path / "log.out"
    p.write_text(text)
    return p


class TestParseTritonLogGpuField:
    def test_absent_gpu_line_yields_none_not_keyerror(self, tmp_path):
        """A log predating the GPU emission must yield None, never raise.

        This is the whole existing corpus. Pre-fix this raises KeyError.
        """
        result = parse_triton_log_file(_write_log(tmp_path, _LOG_WITHOUT_GPU_LINE))
        assert result["gpu"] is None
        # The sibling field must be unaffected -- an anchored CPU pattern selects the
        # same line the unanchored one did.
        assert result["cpu"] == "AMD Ryzen 5 4600H with Radeon Graphics"

    def test_present_gpu_line_is_captured(self, tmp_path):
        result = parse_triton_log_file(_write_log(tmp_path, _LOG_WITH_GPU_LINE))
        assert result["gpu"] == "NVIDIA RTX A6000"
        assert result["cpu"] == "AMD EPYC 7742 64-Core Processor"

    def test_gpu_pattern_does_not_capture_the_two_near_miss_fields(self, tmp_path):
        """`GPUs per task :` and `GPU backend :` must never be read as the device.

        Both clear a naive `GPU\\s*:` by ONE character, so this is a regression guard
        against a future loosening of the pattern -- not a discriminator between the
        anchored and unanchored forms, neither of which matches them today.
        """
        result = parse_triton_log_file(_write_log(tmp_path, _LOG_WITHOUT_GPU_LINE))
        assert result["gpu"] is None
        assert result["gpu_backend"] == "none"
        assert result["gpus_per_task"] == 0

    @pytest.mark.parametrize("path_name", ["does_not_exist.out"])
    def test_missing_file_return_path_carries_the_gpu_key(self, tmp_path, path_name):
        """The missing-file early return is one of THREE all-None dict literals.

        A field added to only some of them fails here rather than in production.
        """
        result = parse_triton_log_file(tmp_path / path_name)
        assert "gpu" in result
        assert result["gpu"] is None

    def test_unparseable_file_return_path_carries_the_gpu_key(self, tmp_path):
        """The `except Exception` fallback is the third literal, and the one a
        synthetic test never otherwise reaches. Reading a DIRECTORY as a file raises
        inside the try, which is what routes execution to the handler."""
        d = tmp_path / "log.out"
        d.mkdir()
        with pytest.warns(UserWarning):
            result = parse_triton_log_file(d)
        assert "gpu" in result
        assert result["gpu"] is None


class TestHardwareLabel:
    def test_known_cpu_string_maps_to_friendly_label(self):
        from hhemt.hardware_labels import hardware_label

        assert hardware_label("AMD EPYC 7742 64-Core Processor") == "epyc-7742"

    def test_known_gpu_string_maps_to_friendly_label(self):
        from hhemt.hardware_labels import hardware_label

        assert hardware_label("NVIDIA RTX A6000") == "a6000"

    def test_unknown_string_returns_the_raw_name_unchanged(self):
        """THE load-bearing case.

        The AMD arm's runtime device string is unverified and unreachable from this
        campaign's cluster, so on Frontier this is the path that fires. Returning a
        shared sentinel ("unknown", None, "") would merge two genuinely different
        unmapped devices into one apparent hardware -- re-creating, inside the feature,
        the heterogeneity blindness the feature exists to remove.

        This assertion discriminates the correct implementation from the two plausible
        wrong ones, not merely from an absent one.
        """
        from hhemt.hardware_labels import hardware_label

        # A deliberately synthetic name, so that GROWING the map can never silently
        # convert this test into a mapped-string test. An input drawn from a real
        # vendor's catalogue would carry exactly that risk.
        raw = "Fictitious Vendor XYZ-9000 Accelerator"
        assert hardware_label(raw) == raw

    def test_map_carries_no_unverified_amd_needle(self):
        """The AMD arm is mapped by NOTHING, deliberately.

        No AMD GPU is reachable from this campaign's cluster, so every candidate
        needle would be a guess at a string nobody has observed. A guessed needle is
        not free: it either silently fails to match (indistinguishable from the
        fallback) or matches something it should not. The fallback already yields the
        correct, if verbose, answer -- so the map stays silent on AMD until a real
        Frontier log supplies the string.

        This pins the DECISION, not a consequence of it, and it is a TRIPWIRE by
        design: it is meant to go red on the day someone adds an AMD needle, so that
        the addition is accompanied by the observed Frontier string that justifies it.
        Deleting this test is the correct way to land that change.

        Asserted BEHAVIOURALLY rather than by reading the map. An earlier draft
        unpacked `_DEVICE_LABELS` as (needle, label) pairs, which made the test fail
        against a correct implementation that stored the map as a dict -- pinning a
        private representation instead of the contract.
        """
        from hhemt.hardware_labels import hardware_label

        for amd in ("AMD Instinct MI250X", "AMD Instinct MI300X"):
            assert hardware_label(amd) == amd

    def test_none_and_blank_stay_absent_rather_than_becoming_a_device(self):
        from hhemt.hardware_labels import hardware_label

        assert hardware_label(None) is None
        assert hardware_label("   ") is None

    def test_matching_is_case_insensitive_and_whitespace_normalised(self):
        from hhemt.hardware_labels import hardware_label

        assert hardware_label("nvidia   rtx a6000") == "a6000"
