#!/usr/bin/env bash
# Source this file to export the same defaults as head2wrist.config.
H2W_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export H2W_REPO_ROOT
export PYTHONPATH="$H2W_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
H2W_PYTHON="${H2W_PYTHON:-$H2W_REPO_ROOT/.venv/bin/python}"
export H2W_PYTHON
eval "$("$H2W_PYTHON" -c 'import shlex; from head2wrist import config; print("\n".join("export "+k+"="+shlex.quote(getattr(config,k)) for k in config.__all__))' )"
