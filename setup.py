"""Setup for ShockArb Factor Model."""

from setuptools import setup, find_packages

setup(
    name="shockarb",
    version="3.0.0",
    description="Geopolitical crisis mispricing detection system",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "scripts*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "yfinance>=0.2",
        "loguru>=0.6",
        "pyarrow>=8.0",
    ],
    extras_require={
        "dev": ["pytest>=7", "pytest-cov"],
    },
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
        "Programming Language :: Python :: 3.12",
    ],
)
