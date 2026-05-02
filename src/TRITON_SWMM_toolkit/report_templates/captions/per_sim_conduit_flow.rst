SWMM conduit flow panel for scenario **{{ snakemake.wildcards.event_id }}**.

Left panel: ratio of maximum flow to full-pipe flow capacity per conduit. Right panel: peak flow magnitude per conduit. Colormaps and bounds are configured in ``report_config.yaml`` under ``per_sim.conduit_flow``.

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
