System map for **{{ snakemake.config['analysis_id'] }}**.

Overlays the watershed polygon (red), TRITON DEM extent (blue dashed), boundary-condition shapefile location (orange markers), and SWMM nodes (black points) and links (gray lines) extracted from a representative scenario via swmmio. The map is rendered in the system CRS configured by ``system_config.crs_epsg``.

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
