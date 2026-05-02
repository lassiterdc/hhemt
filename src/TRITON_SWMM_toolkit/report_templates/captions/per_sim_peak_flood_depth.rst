Peak flood depth raster for scenario **{{ snakemake.wildcards.event_id }}**.

The raster is computed by the TRITON solver as the per-cell maximum H value across all simulation timesteps. Color scale and extent are configured in ``report_config.yaml`` under ``per_sim.peak_flood_depth``.

See `System`_ for the underlying DEM and boundary geometry.

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
