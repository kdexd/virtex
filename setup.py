#!/usr/bin/env python
import glob
import os
from setuptools import setup
import shutil
from typing import List


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over
    these configs inside virtex/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs"
    )
    destination = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "virtex", "model_zoo", "configs"
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if os.path.exists(source_configs_dir):
        if os.path.islink(destination):
            os.unlink(destination)
        elif os.path.isdir(destination):
            shutil.rmtree(destination)

    if not os.path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True)
    return config_paths


setup(
    name="virtex",
    version="1.2.0",
    author="Karan Desai and Justin Johnson",
    description="VirTex: Learning Visual Representations with Textual Annotations",
    package_data={"virtex.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.6",
    license="MIT",
    zip_safe=True,
)
