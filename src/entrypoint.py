from __future__ import annotations

import os
import sys

import uvicorn

from .main import main as run_pipeline


def run() -> None:
    mode = os.getenv("APP_MODE", "api").lower()
    if mode == "api":
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        uvicorn.run("src.api:app", host=host, port=port, reload=False)
        return

    pipeline_args = os.getenv("PIPELINE_MODE", "delta")
    sys.argv = [sys.argv[0], "--mode", pipeline_args]
    run_pipeline()


if __name__ == "__main__":
    run()
