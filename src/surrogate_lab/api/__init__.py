"""FastAPI web layer for surrogate-lab: a report-only demo API.

Deliberately scoped to avoid Vercel's serverless function timeout:

* ``/demo`` serves a pre-trained report baked into the package (no
  training at request time).
* ``/train`` accepts a small user-uploaded CSV and trains *only*
  ``linear_regression`` (the fastest model in the registry), with hard
  caps on upload size and row count, so it stays well under the 10s
  Hobby-plan limit.

Random forest and gradient boosting remain CLI-only; they are never
reachable through this API.
"""
