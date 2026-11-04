"""Executes a Jupyter notebook and writes back the executed outputs."""

import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def execute_notebook(nb_path: Path):
    print(f"Reading {nb_path} ...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    print(f"Executing {nb_path.name} ...")
    client.execute()

    print(f"Saving executed notebook to {nb_path} ...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Completed {nb_path.name} successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        execute_notebook(target)
    else:
        print("Usage: python run_notebook.py <path_to_notebook>")

