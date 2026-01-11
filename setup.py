from setuptools import setup, find_packages

setup(
    name="post-transformer",
    version="0.1.0",
    description="Revolutionary Post-Transformer Intelligence with Active Inference",
    author="MASSIVEMAGNETICS",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    python_requires=">=3.8",
)
