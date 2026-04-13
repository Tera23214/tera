#!/usr/bin/env python
"""
Compatibility entrypoint for the non-uniform N1 loss-vs-step runner.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parent.parent / "loss_vs_step.py"
    spec = importlib.util.spec_from_file_location(
        "_non_uniform_n1_loss_vs_step_main",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load loss-vs-step script from {script_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
