"""Vercel serverless entrypoint.

Vercel's Python runtime auto-detects an ASGI app by scanning `/api` for a
top-level `app` variable. The real implementation lives in
`surrogate_lab.api.main` (see src/surrogate_lab/api/main.py) so the CLI,
local `uvicorn` usage, and this deployment all share the exact same code
path.
"""

from surrogate_lab.api.main import app

__all__ = ["app"]
