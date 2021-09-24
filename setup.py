from os import getenv
from setuptools import find_packages, setup


def install_requires():
    return []


def get_version():
    with open("VERSION") as fp:
        version = fp.read().strip()
    return getenv("PKG_VERSION", version)


def get_description():
    return "Code for working with ensemble members and predicting ensemble members (probabilistic predictions)."


def get_pkg_name():
    return "ensemble"


def get_scripts():
    return []


setup(
    name=get_pkg_name(),
    version=get_version(),
    description=get_description(),
    install_requires=install_requires(),
    author="Steven Brey, Hunter Merrill, Tony Eckel, TCC",
    author_email="steven.brey@climate.com",
    include_package_data=True,
    packages=find_packages(exclude=("test",)),
    scripts=get_scripts(),
    package_data={"ensemble": ["notebooks/*", "docs/resources/*"]},
)
