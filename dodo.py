import itertools
import os
import pathlib
import shlex

from doit.action import CmdAction


HERE = pathlib.Path(__file__).parent
PACKAGE_NAME = "scarce_data_gui"

CI = os.environ.get("CI") == "true"

DOIT_CONFIG = dict(
    verbosity=2,
    backend="json",
    default_tasks=[
        "lint",
    ],
)


def do(*cmd, cwd=HERE):
    if len(cmd) == 1 and callable(cmd[0]):
        cmd = cmd[0]
    else:
        cmd = list(itertools.chain.from_iterable(shlex.split(part) for part in cmd))
    return CmdAction(cmd, shell=False, cwd=cwd)


def task_install():
    """Installs all development requirements and active Learning in development mode"""
    yield dict(
        name="dev",
        file_dep=[HERE / "requirements-dev.txt"],
        actions=[
            do(
                "pip install --upgrade --upgrade-strategy=eager ",
                "-r requirements-dev.txt",
            )
        ],
    )
    yield dict(
        name="project",
        actions=[
            do(
                "pip install .",
            ),
        ],
    )


def task_format():
    """Auto-formats all project files"""
    return dict(
        actions=[
            do("pre-commit run --all-files"),
        ]
    )


def task_lint():
    """Lints all project files"""
    yield dict(
        name="flake8",
        actions=[
            do("flake8 --config=.flake8"),
        ],
    )
    yield dict(
        name="mypy",
        actions=[
            do("mypy --config-file=mypy.ini"),
        ],
    )
