"""A standalone tkinter window showing live Kairnz training-loop status.

Polls the loop's status.json and metrics.jsonl in a work directory and renders
the current iteration/stage, a per-iteration table, and a compact score trend.
"""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from kairnz_train.orchestrate import load_metrics, load_status

# Poll interval in milliseconds.
POLL_MS = 1500


class StatusWindow:
    """Renders training status from a work directory, refreshing on a timer."""

    def __init__(self, root: tk.Tk, work: Path) -> None:
        self.root = root
        self.status_path = work / "status.json"
        self.metrics_path = work / "metrics.jsonl"

        root.title("Kairnz Training")
        root.geometry("520x420")

        self.header = ttk.Label(root, text="waiting for the training loop...", font=("Segoe UI", 12, "bold"))
        self.header.pack(pady=(10, 4), padx=10, anchor="w")

        self.subhead = ttk.Label(root, text="", font=("Segoe UI", 10))
        self.subhead.pack(padx=10, anchor="w")

        columns = ("iter", "samples", "a_score", "promoted")
        self.table = ttk.Treeview(root, columns=columns, show="headings", height=12)
        for col, width in zip(columns, (60, 100, 100, 100)):
            self.table.heading(col, text=col)
            self.table.column(col, width=width, anchor="center")
        self.table.pack(fill="both", expand=True, padx=10, pady=8)

        self.refresh()

    def refresh(self) -> None:
        """Reads the status and metrics files and updates the widgets."""
        status = load_status(self.status_path)
        if status:
            it = status.get("iteration", 0)
            total = status.get("total_iterations", "?")
            stage = status.get("stage", "?")
            promoted = status.get("promoted_count", 0)
            self.header.config(text=f"Iteration {it}/{total}  —  stage: {stage}")
            self.subhead.config(text=f"promoted: {promoted}    last score: {status.get('last_score')}")

        rows = load_metrics(self.metrics_path)
        self.table.delete(*self.table.get_children())
        for row in rows:
            mark = "yes" if row.get("promoted") else "no"
            self.table.insert("", "end", values=(
                row.get("iter"), row.get("samples"),
                f"{row.get('a_score', 0):.3f}", mark,
            ))

        self.root.after(POLL_MS, self.refresh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kairnz training status window.")
    parser.add_argument("--work", type=Path, default=Path("loop-run"), help="Loop working directory.")
    args = parser.parse_args()

    root = tk.Tk()
    StatusWindow(root, args.work.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
