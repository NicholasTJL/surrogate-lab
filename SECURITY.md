# Security Policy

surrogate-lab loads dataset files (CSV/Parquet) and YAML config files, and unpickles saved model
artifacts via `joblib.load`. Treat all three as executable/trusted input: only train on data you
trust, only load model artifacts (`.joblib` files) you trust or produced yourself, and only run
configs from sources you trust. `joblib.load` (like `pickle.load`) can execute arbitrary code if
given a maliciously crafted file.

## Reporting a Vulnerability

If you find a security issue (for example, a way for a crafted dataset or config file to escape
the working directory, leak secrets, or execute unintended code), please report it privately via
GitHub's [private vulnerability reporting](https://github.com/NicholasTJL/surrogate-lab/security/advisories/new)
rather than opening a public issue.

Include:

- A description of the issue and its impact.
- Steps to reproduce, ideally a minimal dataset or config file.
- The surrogate-lab version and operating system.

We aim to acknowledge reports within 5 business days.

## Supported Versions

Only the latest released minor version receives security fixes while the project is pre-1.0.
