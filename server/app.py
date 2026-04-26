"""
server/app.py — OpenEnv multi-mode deployment entry point.

Provides a callable main() entrypoint for graders that launch via
[project.scripts] server = "server.app:main".
"""

import uvicorn
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app

__all__ = ["app"]


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
