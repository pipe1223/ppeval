from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    for _candidate in _HERE.parents:
        if (_candidate / "evaluation" / "__init__.py").exists():
            _ROOT = str(_candidate)
            if _ROOT not in sys.path:
                sys.path.insert(0, _ROOT)
            break



from evaluation.shared.metrics.system import summarize_system_metrics
from evaluation.shared.utils.io import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize system metrics from a list of records.")
    parser.add_argument("json_path", help="Path to a JSON file containing a list of system records.")
    args = parser.parse_args()
    data = load_json(args.json_path)
    if not isinstance(data, list):
        raise TypeError("Expected top-level JSON value to be a list.")
    print(json.dumps(summarize_system_metrics(data), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
