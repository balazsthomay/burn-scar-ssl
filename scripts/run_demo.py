#!/usr/bin/env python3
"""Launch the burn scar segmentation demo server.

Usage:
    uv run scripts/run_demo.py
    uv run scripts/run_demo.py --port 8080
    uv run scripts/run_demo.py --onnx outputs/phase5/full_ft/model_fp32.onnx
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Launch burn scar demo server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--onnx",
        default="outputs/phase5/full_ft/model_fp32.int8.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--data-dir",
        default="data/hls_burn_scars/data",
        help="Path to GeoTIFF data directory",
    )
    parser.add_argument(
        "--results-json",
        default="outputs/phase5/full_ft/deployment_results.json",
        help="Path to deployment results JSON",
    )
    args = parser.parse_args()

    import uvicorn

    from src.demo.app import create_app

    app = create_app(
        onnx_path=args.onnx,
        data_dir=args.data_dir,
        results_json=args.results_json,
    )

    print(f"Starting demo server at http://{args.host}:{args.port}")
    print(f"ONNX model: {args.onnx}")
    print(f"Data dir: {args.data_dir}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
