# cherno

![Versions](https://img.shields.io/badge/python->=3.10-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/sdss-cherno/badge/?version=latest)](https://sdss-cherno.readthedocs.io/en/latest/?badge=latest)
[![Tests Status](https://github.com/sdss/cherno/workflows/Test/badge.svg)](https://github.com/sdss/cherno/actions)
<!-- [![codecov](https://codecov.io/gh/sdss/cherno/branch/main/graph/badge.svg)](https://codecov.io/gh/sdss/cherno) -->


SDSS guider actor

## Installation

In general you should be able to install ``cherno`` by doing

```console
pip install sdss-cherno
```

To build from source, use

```console
git clone git@github.com:sdss/cherno
cd cherno
pip install .
```

## Development

`cherno` uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging. To work with an editable install it's recommended that you setup `uv` and install `cherno` in a virtual environment by doing

```console
uv sync
```

### Style and type checking

This project uses the [black](https://github.com/psf/black) code style with 88-character line lengths for code and docstrings. It is recommended to use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting and the `pyproject.toml` file contains the appropriate configuration. For Visual Studio Code, the following project file is compatible with the project configuration:

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports.ruff": "explicit"
    },
    "editor.wordWrap": "off",
    "editor.tabSize": 4,
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "[markdown]": {
    "editor.wordWrapColumn": 88
  },
  "[restructuredtext]": {
    "editor.wordWrapColumn": 88
  },
  "[json]": {
    "editor.quickSuggestions": {
      "strings": true
    },
    "editor.suggest.insertMode": "replace",
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[yaml]": {
    "editor.insertSpaces": true,
    "editor.formatOnSave": true,
    "editor.tabSize": 2,
    "editor.autoIndent": "advanced",
  },
  "prettier.tabWidth": 2,
  "editor.rulers": [88],
  "editor.wordWrapColumn": 88,
  "python.analysis.typeCheckingMode": "basic",
  "ruff.nativeServer": true
}
```

This assumes that the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) extensions are installed.
