# cherno

![Versions](https://img.shields.io/badge/python->3.8-blue)
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

`cherno` uses [poetry](http://poetry.eustace.io/) for dependency management and packaging. To work with an editable install it's recommended that you setup `poetry` and install `cherno` in a virtual environment by doing

```console
poetry install
```

### Style and type checking

This project uses the [black](https://github.com/psf/black) code style with 88-character line lengths for code and docstrings. It is recommended that you run `black` on save. Imports must be sorted using [isort](https://pycqa.github.io/isort/). The GitHub test workflow checks all the Python file to make sure they comply with the black formatting.

Configuration files for [flake8](https://flake8.pycqa.org/en/latest/), [isort](https://pycqa.github.io/isort/), and [black](https://github.com/psf/black) are provided and will be applied by most editors. For Visual Studio Code, the following project file is compatible with the project configuration:

```json
{
    "python.formatting.provider": "black",
    "[python]" : {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        },
        "editor.formatOnSave": true
    },
    "[markdown]": {
        "editor.wordWrapColumn": 88
    },
    "[restructuredtext]": {
        "editor.wordWrapColumn": 88
    },
    "editor.rulers": [88],
    "editor.wordWrapColumn": 88,
    "python.analysis.typeCheckingMode": "basic"
}
```

This assumes that the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) extensions are installed.

This project uses [type hints](https://docs.python.org/3/library/typing.html). Typing is enforced by the test workflow using [pyright](https://github.com/microsoft/pyright) (in practice this means that if ``Pylance`` doesn't produce any errors in basic mode, ``pyright`` shouldn't).
