from __future__ import annotations

import json

from .pipeline import run_project_pipeline
from .reporting import generate_report


def main() -> None:
    results = run_project_pipeline()
    report_paths = generate_report(results)

    summary = {
        "comparison_path": str(results["comparison_path"]),
        "roc_curve_path": str(results["roc_curve_path"]),
        "report_markdown": str(report_paths["markdown"]),
        "report_docx": str(report_paths["docx"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
