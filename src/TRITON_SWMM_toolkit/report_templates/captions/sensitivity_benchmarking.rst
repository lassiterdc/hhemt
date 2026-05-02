Benchmarking-mode sensitivity plot for independent variable **{{ snakemake.wildcards.independent_var }}**.

The dependent variable (default ``performance.Total`` from the per-scenario performance summary, sum-across-timesteps mean-across-MPI-ranks, restart-safe) is aggregated per sub-analysis using the ``aggregation`` mode declared in ``report_config.yaml`` under ``sensitivity``. SWMM-only sub-analyses route to ``swmm_full_rpt_file`` Total elapsed time (parsed via ``swmm_output_parser.parse_total_elapsed``).

**Sources:**

{% for src in snakemake.params.source_paths %}
{%- if src is mapping %}
- ``{{ src.path }}``
{%- if src.variables %}
{%- for v in src.variables %}
    - ``{{ v }}``
{%- endfor %}
{%- endif %}
{%- else %}
- ``{{ src }}``
{%- endif %}
{% endfor %}
