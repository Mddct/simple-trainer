import platform
from setuptools import setup, find_packages

requirements = [
    "ml_collections", "tensorboard", "wrapt", "etils", "importlib_resources"
]


if platform.system() == 'Windows':
    requirements += ['PySoundFile']

setup(
    name="simple-trainer",
    install_requires=requirements,
    packages=find_packages(),
)
