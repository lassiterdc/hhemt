Aggregate scenario status and continuity error metrics for **{{ snakemake.config['analysis_id'] }}**. The SWMM continuity error reported is the *flow routing* continuity (not the runoff-quantity continuity) parsed from each scenario's SWMM ``.rpt`` via ``swmm_output_parser.return_swmm_system_outputs``. SWMM considers continuity errors above ~10% grounds to question simulation validity (SWMM User's Manual v5.2 §8.5 Excessive Continuity Errors). Data source paths below are relative to the analysis directory root; ``.rpt`` files are SWMM Status Reports written at simulation end.

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
