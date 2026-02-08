import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from TRITON_SWMM_toolkit import run_model
from pathlib import Path
import yaml


def launch_gui():
    # Use TkinterDnD for drag-and-drop
    root = TkinterDnD.Tk()
    root.title("TRITON SWMM Toolkit")
    root.geometry("600x300")

    # ------------------------------
    # Config file input
    # ------------------------------
    tk.Label(root, text="Config File:").pack(pady=(10, 0))
    entry_path = tk.Entry(root, width=60)
    entry_path.pack(padx=10, pady=5)

    def browse_file():
        path = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if path:
            entry_path.delete(0, tk.END)
            entry_path.insert(0, path)

    tk.Button(root, text="Browse", command=browse_file).pack(pady=(0, 5))

    # Enable drag-and-drop
    def drop(event):
        files = root.tk.splitlist(event.data)
        if files:
            entry_path.delete(0, tk.END)
            entry_path.insert(0, files[0])  # only take first file

    entry_path.drop_target_register(DND_FILES)  # type: ignore
    entry_path.dnd_bind("<<Drop>>", drop)  # type: ignore

    # ------------------------------
    # Additional simulation options
    # ------------------------------
    verbose_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Verbose", variable=verbose_var).pack(pady=5)

    # Example: numeric setting (e.g., timestep)
    tk.Label(root, text="Time Step (minutes):").pack(pady=(10, 0))
    entry_timestep = tk.Entry(root, width=10)
    entry_timestep.insert(0, "5")
    entry_timestep.pack()

    # ------------------------------
    # Run button
    # ------------------------------
    def run():
        config_path = entry_path.get()
        verbose = verbose_var.get()
        timestep = entry_timestep.get()

        if not config_path:
            messagebox.showwarning("Missing file", "Please select a config file!")
            return
        try:
            # Convert timestep to int
            timestep = int(timestep)
            # Call core logic
            run_model(config_path=config_path, verbose=verbose, timestep=timestep)
            messagebox.showinfo("Done", "Simulation finished successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(root, text="Run Simulation", command=run, bg="green", fg="white").pack(
        pady=15
    )

    root.mainloop()
