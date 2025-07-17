"""Setup module."""

import json
import os
import re
from datetime import datetime

from setuptools import find_namespace_packages, setup


def load_extra_requirements() -> dict:
    """Load extra requirements from "extra_requirements.json".

    Eg to run locally: >>> pip install -e ".[holidays]"

    Returns (dict(str, list(str)):
        Eg:
        {'cpu-info': ['py-cpuinfo'],
         'holidays': ['holidays>=0.26'],
         'pandas': ['pandas>=1.2.5']}
    """
    with open("extra_requirements.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    extra = {k: v["packages"] for k, v in json_data.items()}

    # Add the "full" option to install all dependencies
    extra.update({"full": list({i for ii in extra.values() for i in ii})})
    return extra


def make_version() -> str:
    """Construct the version string.

    https://www.python.org/dev/peps/pep-0440/.

    Returns (str):
        A str which represent the current version of the library.
    """
    # information on current git branch
    branch = os.getenv("CI_COMMIT_REF_NAME", "local")
    branch_no = re.fullmatch(pattern=r"(?P<number>\d+)(?P<name>-\S*)", string=branch)

    # public version identifier, using only:
    # - release segment (date based)
    # - development release segment
    rs = f"{datetime.now().strftime('%Y.%m.%d.%H.%M')}"
    ds = f".dev{'' if branch_no is None else branch_no.group('number')}"
    pvi = f"{rs}{ds if branch not in ['main', 'master', 'develop'] else ''}"

    # local version identifier  = <public version identifier>[+<local version label>]
    return f"{pvi}+local" if branch in ["local"] else pvi


version = make_version()

print(f"Version LIVE: {version}")

setup(
    name="nlsnnebula",
    version=version,
    packages=find_namespace_packages(include=["nlsn.*", "nlsn"]),
    namespace_packages=["nlsn"],
    python_requires="~=3.7",
    extras_require=load_extra_requirements(),
    # metadata to display on PyPI
    author="DSci Global Platform Support Team",
    author_email="dscigps@nielsen.com",
    description="Nebula: A collection of pyspark transformers",
    keywords="nielsen",
    url="http://www.nielsen.com",  # project home page, if any
    project_urls={
        "Source Code": "https://gitlab.com/nielsen-media/dsci/dsciint/dsci-gps/nebula",
    },
    classifiers=["License :: Proprietary"],
)
