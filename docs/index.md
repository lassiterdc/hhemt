# H&H Ensemble Modeling Toolkit

The H&H Ensemble Modeling Toolkit, hhemt, orchestrates coupled TRITON-SWMM flood
simulations as ensembles across workstations and HPC clusters. The toolkit manages the
full lifecycle from preprocessing through post-processing and produces consolidated
datasets and an interactive report.

--8<-- "platform-support.md"

## Highlights

- **Open science.** [Bundles and publishing](how-to/publishing.md) package results with their metadata for deposit in Zenodo or HydroShare.
- **Selective reruns.** The Snakemake workflow manager [reruns only the simulations whose inputs changed](how-to/forcing-reruns.md).
- **Limited peak disk use.** The toolkit processes outputs as simulations finish, and the [`clear_raw` option](reference/config-schema.md) deletes the raw outputs afterward.
- **Interactive reports.** The toolkit renders [a self-contained report](reference/example-report.md) for each completed analysis.
- **Exploratory analysis.** The `hhemt eda` command [writes a Jupyter notebook](tutorials/exploratory-analysis.md) for continued work on a result.
- **Metadata tracking.** Consolidated outputs record the toolkit version, the provenance, and the CF-1.13 metadata listed in the [output data model](reference/output-data-model.md).
- **Containers.** [Container mode](how-to/building-container-images.md) packages the solver in an Apptainer image and executes each simulation in that image.
- **Parallel simulation.** Simulations run on CPU or on NVIDIA and AMD GPUs across workstations and [SLURM clusters](explanation/capabilities.md#hardware-portability).

## License

The source is available on [GitHub](https://github.com/lassiterdc/hhemt) and on PyPI through `pip install hhemt`.
The license is the [GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.html).
The first release was published under the PolyForm Noncommercial License 1.0.0.

<a id="how-to-cite"></a>

## Citation

If you use hhemt in published work, cite the software.

``` { .text .copy }
Lassiter, D. (2026). H&H Ensemble Modeling Toolkit (hhemt) (Version 0.2.1)
[Computer software]. https://doi.org/10.5281/zenodo.21359151
```

<!-- hhemt:doi-kind-convention
     A page built from a release tag names only identifiers that exist at that tag.
     Zenodo mints the version DOI after the release is published, so the block above
     carries the CONCEPT DOI. The version DOI of the release is added to the `latest`
     build of this page after the tag, in a docs-only change. CITATION.cff carries the
     CONCEPT DOI for the same reason. Reasoning and worked cases are in
     architecture.md. -->

Zenodo issues a *concept* DOI and a *version* DOI. The concept DOI resolves to the
newest release. A version DOI resolves to a single release permanently. Zenodo creates the
version DOI after a release is published, so the citation carries the concept DOI. To
pin the exact version of a result, take the version DOI from the Zenodo record that the
concept DOI resolves to. The DOI badge on the repository page and the "Cite this
repository" button hand out the concept DOI. A consolidated output records the toolkit
version that produced the output.

| DOI | Resolves to | Use case |
|---|---|---|
| `10.5281/zenodo.21359151` | always the latest version | Pointing a reader at the software in general. |

`CITATION.cff` in the repository root carries the same metadata in a
machine-readable form, and GitHub renders the file as a "Cite this repository" button.

## Next steps

- The [Quickstart](quickstart.md) starts from a fresh clone and ends with a running simulation.
- [Installation](how-to/installation.md) covers the environment setup and the pip-only alternative.
- [Tutorials](tutorials/index.md) walk through complete case studies.
- [How-To Guides](how-to/index.md) cover single tasks, from configuring an analysis to publishing a dataset.
- [Reference](reference/index.md) lists the API, the configuration schema, and the outputs.
- [Limitations](explanation/limitations.md) covers the solver, environment, scale, and reproducibility constraints.
