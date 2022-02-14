#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("LaMer", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


setup(
    name="LaMer",
    version=get_version(),
    description="A Text Style Transfer Framework",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DapangLiu/LaMer",
    author="Ruibo Liu",
    author_email="ruibo.liu.gr@dartmouth.edu",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="text style transfer reinforcement learning",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=[
        "tqdm",
        "numpy>1.16.0",
        "pandas",
        "scikit-learn",
        "transformers",
        "torch>=1.4.0",
        "sentence_transformers",
        "spacy",
    ],
    extras_require={
        "dev": [
            "sphinx<4",
            "sphinx_rtd_theme",
            "sphinxcontrib-bibtex",
            "flake8",
            "flake8-bugbear",
            "yapf",
            "isort",
            "pytest",
            "pytest-cov",
            "wandb>=0.12.0",
            "mypy",
            "pydocstyle",
            "doc8",
            "scipy",
        ],
    },
)
