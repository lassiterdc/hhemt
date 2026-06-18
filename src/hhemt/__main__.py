import sys
from .cli import app
from .gui import launch_gui

if __name__ == "__main__":
    # If no args, launch GUI
    if len(sys.argv) == 1:
        launch_gui()
    else:
        app()
