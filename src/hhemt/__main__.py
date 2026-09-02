import sys

from .cli import app

if __name__ == "__main__":
    # If no args, launch GUI
    if len(sys.argv) == 1:
        # Imported HERE, not at module level: the GUI stack (tkinterdnd2) is an
        # optional desktop dependency, and an eager import made every CLI
        # invocation through `python -m hhemt` fail with ModuleNotFoundError in
        # any environment without it. The call site was always guarded; only the
        # import was not.
        from .gui import launch_gui

        launch_gui()
    else:
        app()
