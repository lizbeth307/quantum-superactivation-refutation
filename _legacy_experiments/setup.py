from setuptools import setup, find_packages

setup(
    name="quantum_core",
    version="2.0.0",
    description="Quantum Superactivation Engine Core Library",
    author="Quantum Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0"
    ],
    python_requires=">=3.8",
)
