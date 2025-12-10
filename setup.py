"""
Setup configuration for loadcell-temp-compensation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loadcell-temp-compensation",
    version="1.0.0",
    author="Jueseok Kim",
    author_email="kimjue@mail.uc.edu",
    description="Temperature compensation for load cell measurements using segment-based linear regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kim5jk/loadcell-temp-compensation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "viz": ["plotly>=5.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    keywords=[
        "load cell",
        "temperature compensation",
        "sensor drift",
        "signal processing",
        "measurement",
        "calibration",
    ],
)
