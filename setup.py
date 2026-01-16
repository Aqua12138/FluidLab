#!/usr/bin/env python3
# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent 安装脚本"""

from setuptools import setup, find_packages

setup(
    name="rheo",
    version="1.0.0",
    author="RheoAgent Team",
    description="RheoAgent - 基于 MPM 的流体仿真与强化学习框架",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rheo/rheo",
    license="BSD-3-Clause",
    
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    
    python_requires=">=3.8",
    
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "taichi>=1.5.0",
        "gymnasium>=0.28.0",
        "yacs>=0.1.8",
        "scipy>=1.7.0",
        "trimesh>=3.9.0",
        "pyyaml>=5.4.0",
    ],
    
    extras_require={
        "rl": [
            "skrl>=1.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "rheo-keyboard=script.keyboard:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    
    keywords="fluid simulation, MPM, reinforcement learning, differentiable physics",
)
