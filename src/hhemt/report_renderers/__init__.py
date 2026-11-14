"""Report renderer modules. Each module exposes `render(analysis, report_cfg, output_path) -> Path`.

This package re-exports nothing and imports nothing from `hhemt`, and both facts are load-bearing.
`hhemt.config.report` reaches `report_renderers._reporting_sets` lazily, and a renderer such as
`workflow_performance` imports `hhemt.config.report`; an import added here would be executed by
every one of those paths and close the `config.report -> report_renderers -> config.report` cycle.
The public API of each renderer is the `__all__` of its own module, so the package itself declares
an empty one.
"""

__all__: list[str] = []
