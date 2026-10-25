# Software availability and reproducibility

The H&H Ensemble Modeling Toolkit (hhemt) is publicly available at <https://github.com/lassiterdc/hhemt> and on PyPI (`pip install hhemt`).

**License.** The source is licensed under the [GNU Affero General Public License, version 3 or any later version](https://www.gnu.org/licenses/agpl-3.0.html) (AGPL-3.0-or-later; see `LICENSE`). Release 0.1.0, on PyPI and Zenodo, was published under the PolyForm Noncommercial License 1.0.0.

**How to cite.** Cite the software via its Zenodo DOI (see the repository's "How to cite" section and `CITATION.cff`).

**Reproducibility.** The environment is reproduced from `environment.yaml` (`conda env create`); the toolkit installs on top via `pip install -e .`. Two test tiers guard regressions: a fast synthetic tier and a real-data tier built on the Norfolk, VA case study.

**Data availability.** The Norfolk case-study inputs are published on HydroShare and download anonymously via the toolkit's integrated `case.yaml` path.

**Reproduce the release.** Install → get the Norfolk data → run a predefined experiment → smoke-test with `analysis.test()`. See the [Norfolk end-to-end tutorial](../tutorials/norfolk-end-to-end.md).
