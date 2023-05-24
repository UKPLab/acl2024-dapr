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
        "colbert @ git+https://github.com/stanford-futuredata/ColBERT.git",
        "ujson",
        "wandb",
        "datasets",
        "more_itertools",
        "pytrec_eval",
        # "gitpython",
        # "psutil",
        # "aiohttp",
        # "scipy",
        "sentence-transformers",
        "hydra-core",
        "pke @ git+https://github.com/boudinfl/pke.git",
        "gdown",
        "ir_datasets",
        "pyserini"
    ],
)
