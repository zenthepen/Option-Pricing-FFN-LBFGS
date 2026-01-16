from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="double-heston-lbfgs",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Double Heston Model with Jumps: L-BFGS Calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Option-Pricing-FFN-LBFGS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "ffn": [
            "tensorflow>=2.8.0",
        ],
    },
)
