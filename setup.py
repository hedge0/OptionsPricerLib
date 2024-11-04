from setuptools import setup, find_packages

setup(
    name="OptionsPricerLib",
    version="0.1.0",
    description="A library for pricing options using different models",
    author="hedge0",
    packages=find_packages(),
    install_requires=["numpy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
