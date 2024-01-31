from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="dapr",
    version="0.0.0",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="A benchmark on Document-Aware Passage Retrieval (DAPR).",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://https://github.com/kwang2049/dapr",
    project_urls={
        "Bug Tracker": "https://github.com/kwang2049/dapr/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "colbert @ git+https://github.com/stanford-futuredata/ColBERT.git@21b460a606bed606e8a7fa105ada36b18e8084ec",
        "ujson==5.7.0",
        "datasets==2.16.1",
        "more-itertools==9.1.0",
        "matplotlib==3.7.4",
        "pytrec-eval==0.5",
        "transformers==4.37.2",
        "pke @ git+https://github.com/boudinfl/pke.git",
        "ir-datasets==0.5.4",
        "pyserini==0.21.0",
        "clddp==0.0.8",
    ],
)
