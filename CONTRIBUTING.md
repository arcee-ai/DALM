# Contributing to DALM

Thanks for helping out! We're excited for your issues and PRs

## Building from local

Building the repo is straightforward. Clone the repo, and install the package. We use [invoke](https://github.com/pyinvoke/invoke) to manage `DALM`
```shell
git clone https://github.com/arcee-ai/DALM.git && cd DALM
pip install invoke
inv install
```
This will install the repo, with its dev dependencies, in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (for live updates on code changes)

## Format, lint, test
Because we use `invoke`, the following is all you need to prepare for a pr
```shell
inv format  # black, ruff
inv lint    # black check, ruff check, mypy
inv test    # pytest
```

We require 95% test coverage for all PRs.

For more information around our `invoke` commands, see [`tasks.py`](https://github.com/arcee-ai/DALM/blob/main/tasks.py) and our [`pyproject.toml`](https://github.com/arcee-ai/DALM/blob/main/pyproject.toml) configuration