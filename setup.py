from setuptools import setup, find_packages

setup(
    name="acia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0"
    ],
    author="Arman Behnam",
    description="Anti-Causal Invariant Abstractions for OOD Generalization",
    python_requires=">=3.7"
)