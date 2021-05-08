from pathlib import Path
from setuptools import setup


HERE = Path(__file__).parent

README = Path(HERE, "README.md").read_text()

setup(
    name="nnvis",
    version="0.0.1",
    description="Visualizing tool for PyTorch NN models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Silvie Němcová",
    author_email="nemcova.silva@gmail.com",
    license="MIT",
    packages=["nnvis"],
    install_requires=["numpy", "torch", "torchvision", "scipy", "sklearn", "matplotlib", "h5py"]
)