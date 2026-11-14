"""The email pattern reads a non-ASCII local part, a non-ASCII domain label and a non-ASCII top-level label.

Package 1e, chunk 1e.1 (approved shape S1 D1). The anonymization gate (`hhemt.bundle._reprex_gate`) and
the repository guard (`scripts/check_anonymization.py`) each carry one email pattern, kept equal by a pin in
`tests/test_bundle_emit_refuses_blocklisted.py`. The pattern was ASCII-only, so an address whose local part
or domain is written in another script (an accented local part, an internationalised domain, an
internationalised top-level label) passed both scans and reached a published bundle.

SOLVER-FREE: plain strings and the two compiled patterns. Every non-ASCII character is written as a
``\\u`` escape and every address is built from fragments, so this file is itself clean under the repository
guard (no exemption row covers a test file).

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What
kills it, and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import pytest

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest
from hhemt.bundle import _reprex_gate as G

AT = "@"
E_ACUTE = "é"  # a Latin letter outside ASCII
U_UMLAUT = "ü"
HAN_ZHONG, HAN_GUO = "中", "国"  # a two-letter top-level label in Han script
CYRILLIC_PRIMER = "пример"  # a Cyrillic domain label


def _address(local: str, domain: str) -> str:
    return local + AT + domain


#: name -> an address whose text a reader can recover and whose scan must NOT pass. Each row varies
#: WHICH part of the address is non-ASCII, which is the property the old pattern could not read.
NON_ASCII_ADDRESSES = {
    "local-part": _address("jos" + E_ACUTE + ".garcia", "mail.example.com"),
    "local-part-leading": _address(E_ACUTE + "lodie", "mail.example.com"),
    "domain-label": _address("someone", "b" + U_UMLAUT + "cher.example.de"),
    "domain-label-cyrillic": _address("someone", CYRILLIC_PRIMER + ".example.com"),
    "top-level-label": _address("someone", "mail.example." + HAN_ZHONG + HAN_GUO),
    "all-three": _address("jos" + E_ACUTE, "b" + U_UMLAUT + "cher." + HAN_ZHONG + HAN_GUO),
}

#: Strings that are NOT addresses; the pattern must stay silent on every one (the false-positive arm).
NOT_ADDRESSES = {
    "at-with-spaces": "meet at caf" + E_ACUTE + " " + AT + " 5pm",
    "numeric-tld": "v1" + AT + "host.12",
    "underscore-tld": "name" + AT + "host.a_b",
    "no-local-part": AT + "b" + U_UMLAUT + "cher.example.de",
    "single-letter-tld": "name" + AT + "host.a",
    "ascii-handle-no-domain": "see " + AT + "someone on the thread",
}


def _gate_hits(text: str) -> list[object]:
    return G.ZeroUserInfoScanner.for_consume().scan_member("m", text.encode("utf-8"))


@pytest.mark.parametrize("name", sorted(NON_ASCII_ADDRESSES))
def test_both_patterns_match_the_whole_non_ascii_address(name):
    """Property: each pattern matches the ENTIRE address (not an ASCII tail of it) wherever the address sits in
        a line, for a non-ASCII local part, domain label and top-level label.

    Class: REGRESSION - red on the pre-change tree: the ASCII-only classes find no match at all for an
        accented local part and only a tail (or nothing) for a non-ASCII domain, output pasted in the return.
    Reachability: a researcher's contact address in a config or a licence header is not restricted to ASCII;
        the gate reads the text members of every emitted bundle and the guard reads every tracked file.
    Both-states anchor: the assertion turns on the matched span equalling the address, a property both the
        old and the new pattern have (the old one returns a shorter span or none).
    Invocation isolated: pattern objects only; no path, environment variable or working directory is read.
    What kills it: leaving either pattern ASCII-only, or widening only one of the three parts.
    Second correct implementation: a pattern built from a Unicode-aware alphabetic test per part instead of
        `\\w`, with the same spans.
    """
    address = NON_ASCII_ADDRESSES[name]
    line = "contact: " + address + " (maintainer)"
    for label, pattern in (("gate", G.EMAIL_PATTERN), ("guard", guard.EMAIL_RE)):
        match = pattern.search(line)
        assert match is not None, f"{label} pattern did not fire on the {name} address"
        assert match.group(0) == address, f"{label} pattern matched only {match.group(0)!r} of the {name} address"


@pytest.mark.parametrize("name", sorted(NON_ASCII_ADDRESSES))
def test_the_gate_scanner_and_the_guard_refuse_a_non_ascii_address(name):
    """Property: the scanner the emit runs, and the guard's per-line class check, each report the email class
        for a line carrying a non-ASCII address and no other producer-shaped value.

    Class: REGRESSION - red on the pre-change tree for the rows with no ASCII residue the old pattern can match
        (all-three, domain-label, domain-label-cyrillic); the other three rows fire on the old pattern through
        an ASCII tail of the address, so they are green before the change and the span test above is what
        reds them. Output pasted in the return.
    Reachability: `ZeroUserInfoScanner.scan_member` and `pattern_class_hits` are the two call sites of the
        two patterns; this runs both on the whole path from text to verdict.
    Both-states anchor: the verdict is the email class firing, which the old code also returns for an ASCII
        address (the control test below).
    Invocation isolated: the scanner is the consume-side one (no config, no private list); the guard call
        takes an empty exempt set.
    What kills it: a scanner that decodes members with `errors="ignore"` and drops the non-ASCII character
        (the address then reads as its ASCII residue), or a guard that lowercases through an ASCII table.
    Second correct implementation: the scanner matching on bytes with a UTF-8 aware pattern.
    """
    line = "contact: " + NON_ASCII_ADDRESSES[name]
    assert [h.detail.split(" ")[0] for h in _gate_hits(line)] == ["email-address"], name
    assert guard.pattern_class_hits(line, frozenset({guard.CLASS_EMAIL}), frozenset()) == [guard.CLASS_EMAIL], name


def test_the_widened_pattern_keeps_every_ascii_verdict_and_stays_silent_on_non_addresses():
    """Property: an ASCII address still fires both patterns with the same span, and none of the strings that
        merely resemble an address fires either pattern (the false-positive arm of the widening).

    Class: NEW CAPABILITY - the plausible wrong implementation is `.` for the local part and domain labels
        (match anything), which fires on every NOT_ADDRESSES row, and the numeric-top-level row is red under
        a TLD class of `\\w{2,}` (digits and underscore admitted).
    Reachability: the two patterns scan every tracked file and every bundle member, where a handle with an
        at-sign (a social handle, a decorator) is common.
    Both-states anchor: the silent rows contain an at-sign and Unicode text in both the old and the new tree.
    Invocation isolated: pattern objects only.
    What kills it: over-widening the domain or top-level class; dropping the look-behind or the minimum
        two-letter top-level label.
    Second correct implementation: a stricter per-label grammar (no leading or trailing hyphen), which still
        passes every row here.
    """
    ascii_address = _address("a.b+c", "mail.example.com")
    for label, pattern in (("gate", G.EMAIL_PATTERN), ("guard", guard.EMAIL_RE)):
        match = pattern.search("x " + ascii_address + " y")
        assert match is not None and match.group(0) == ascii_address, label
        for name, text in NOT_ADDRESSES.items():
            assert pattern.search(text) is None, f"{label} pattern fired on the non-address {name}"


def test_the_gate_and_the_guard_still_hold_one_pattern_string():
    """Property: the gate's pattern string equals the guard's, so the two scans cannot disagree on an address.

    Class: REGRESSION - the pin in `tests/test_bundle_emit_refuses_blocklisted.py` already holds; this
        restates it beside the widening so a half-edit fails in the file that owns the widening.
    Reachability: the two definitions live in files that import nothing from each other.
    Both-states anchor: equality of the strings is a property of both trees' definitions.
    Invocation isolated: module attributes only.
    What kills it: editing either email pattern alone.
    Second correct implementation: the gate importing the guard's string (the guard must stay free of
        toolkit imports, so the reverse direction is the only one and it is not taken).
    """
    assert G.EMAIL_PATTERN.pattern == guard.EMAIL_RE.pattern
