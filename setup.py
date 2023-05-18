import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LoRA",
    version="0.1.0",
    description="PyTorch integration of Low-Rank Adaptation (LoRA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kim Roder",
    author_email="kim.roder@gmail.com",
    url="https://github.com/TheDiscoMole/LoRA",
    python_requires='>=3.10',
    install_requires=[
        "torch"
    ],
    packages=setuptools.find_packages(),
)
