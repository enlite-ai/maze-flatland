"""
The setup.py script used to install this project a pip module
"""
from __future__ import annotations

from setuptools import find_namespace_packages, setup

setup(
    name='maze_flatland',
    packages=find_namespace_packages(include=['maze_flatland', 'maze_flatland.*', 'hydra_plugins']),
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml'],
    },
    install_requires=[],
)
