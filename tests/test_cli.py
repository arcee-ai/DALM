import os

from dalm import __version__


def test_cli_version() -> None:
    version_msg = os.popen("dalm version").readlines()[-1].strip()
    assert version_msg == f"🐾You are running DALM version: {__version__}"
