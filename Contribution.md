# Contributing guide lines

We appreciate all contributions! If you are planning to contribute bug-fixes or
documentation improvements, please go ahead and open a
[pull request (PR)](https://github.com/jbueltemeier/scarce_data_gui/-/merge_requests)
. If you are planning to contribute new features, please open an
[issue](https://github.com/jbueltemeier/scarce_data_gui/-/issues) and
discuss the feature with us first.

To start working on `scarce_data_gui` clone the repository from GitHub and set up
the development environment

```shell
git clone https://github.com/jbueltemeier/scarce_data_gui
cd scarce_data_gui
python -m pip install --user virtualenv (if not installed)
virtualenv .venv --prompt='(scarce_data_gui-dev) '
source .venv/bin/activate (on Linux) or .venv\Scripts\activate (on Windows)
pip install doit
doit install
```

Every PR is subjected to multiple checks that it has to pass before it can be merged.
The checks are performed through [doit](https://pydoit.org/). Below you can find details
and instructions how to run the checks locally.

## Debug with IDE

You have to setup the Run/Debug config to debug the streamlit app. (See:
[Help](https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide)
)

```shell
On PyCharm 'Run' -> 'Edit Configurations...'
Add new configuration
Python
Use a Name for the configuration (Streamlit GUI)
'Module name': streamlit
'Parameters': run Home.py
Set Working directory:
For Example: 'C:/Path_TO_Repo/scarce_data_gui'
```

## Code format and linting

`scarce_data_gui` uses [ufmt](https://ufmt.omnilib.dev/en/stable/) to format
Python code, and [flake8](https://flake8.pycqa.org/en/stable/) to enforce
[PEP8](https://www.python.org/dev/peps/pep-0008/) compliance.

Furthermore, `scarce_data_gui` is
[PEP561](https://www.python.org/dev/peps/pep-0561/) compliant and checks the type
annotations with [mypy](http://mypy-lang.org/) .

To automatically format the code, run

```shell
doit format
```

Instead of running the formatting manually, you can also add
[pre-commit](https://pre-commit.com/) hooks. By running

```shell
pre-commit install
```

once, an equivalent of `doit format` is run everytime you `git commit` something.

Everything that cannot be fixed automatically, can be checked with

```shell
doit lint
```
