# Explanation

Understanding-oriented discussion: the concepts, capabilities, and design
rationale behind the toolkit.

- [Capabilities](capabilities.md): what the toolkit does and the study
  designs it supports.
- [Limitations and constraints](limitations.md): what the toolkit does not
  contain, what it constrains about your environment, and what it does not
  claim about scientific validity.
- [Coupled inflow nodes](coupled-inflow-nodes.md): which SWMM nodes exchange
  water with the surface, and what each preparation error about them means.
- [Software availability](software-availability.md): licensing and how to
  obtain the coupled solvers.
- [When and why re-runs happen](rerun-faq.md): the Snakemake re-run triggers
  and the cases that surprise people.
- [Branching and release model](branching-and-release-model.md): the
  `develop`/`main` gitflow-lite model.
- [Exploratory analysis](exploratory-analysis.md): what the post-run
  exploratory step is for, and why it sits outside the workflow graph.
- [Testing the toolkit](testing-the-toolkit.md): why the suite has two tiers, why a
  fixture rebuild is silent, and what a green claim covers.
