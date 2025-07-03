from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def read_requirements():
    """Read requirements from requirements.txt file."""
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="lorenz-attractor-pro",
    version="2.0.0",
    author="Jose Solis Rosales",
    author_email="josesolisrosales@linux.com",
    description="Professional-grade Lorenz attractor simulation suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josesolisrosales/lorenz-attractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0"],
        "gpu": ["cupy>=12.0.0", "cuda-python>=12.0.0"],
    },
    entry_points={
        "console_scripts": [
            "lorenz-sim=lorenz_attractor.cli:main",
            "lorenz-web=lorenz_attractor.web.app:main",
        ],
    },
)