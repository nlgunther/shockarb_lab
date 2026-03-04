"""Setup for ShockArb Factor Model."""

from setuptools import setup, find_packages

setup(
    name="shockarb",
    version="2.1.0",
    description="Geopolitical crisis mispricing detection system",
    author="ShockArb Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "yfinance>=0.2",
        "loguru>=0.6",
        "pyarrow>=8.0",
    ],
    entry_points={
        "console_scripts": [
            "shockarb=shockarb.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
