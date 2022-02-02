from pathlib import Path
from setuptools import setup


HERE = Path(__file__).parent

README = Path(HERE, "README.md").read_text()

setup(
    name="nnvis",
    version="1.1.4",
    description="Visualizing tool for PyTorch NN models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Silvie Němcová",
    author_email="nemcova.silva@gmail.com",
    license="MIT",
    packages=["nnvis"],
    install_requires=["numpy", "torch>=1.8.1", "torchvision>=0.9.1",
                      "scipy", "sklearn", "matplotlib", "h5py", "tqdm"],
    project_urls={
        "Source": "https://github.com/suzrz/nnvis",
        "Tracker": "https://github.com/suzrz/nnvis/issues"
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",
        "Environment :: MacOS X",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Utilities"
    ]
)
