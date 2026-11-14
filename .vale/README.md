# Vale styles

This directory is the one source of the four Vale styles that lint the hhemt documentation. The hhemt repository maintains them, and no other copy is maintained. A copy of these files in another repository is a snapshot that nothing keeps current, and the rules here win over it.

| Style | What it enforces |
| --- | --- |
| `HhemtDocs` | The prose conventions of the documentation pages: banned phrases, label and signpost patterns, link text, and the Linux wording rules. |
| `HhemtQuality` | The quality rules that apply to every page, such as heading case, sentence length, paragraph length, and spelling. |
| `HhemtMarkers` | The rules against text that addresses an AI assistant, capitalised emphasis words, and status emoji. |
| `HhemtContributing` | The extra rules that apply only to `docs/contributing.md`. |

`.vale.ini` at the repository root selects the styles for each page. To check the pages, run this from the repository root:

```bash
python scripts/check_vale.py
```

The script fails on any alert, because every rule has warning level and Vale itself exits 0 on a warning.

To change a rule, edit the file here and run the script in the same commit, so the pages and the rule stay in step. Do not copy a style file to another location to edit it there.
