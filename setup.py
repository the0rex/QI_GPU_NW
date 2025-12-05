from setuptools import setup, find_packages

setup(
    name="qi-align",
    version="1.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "cupy-cuda12x",
        "pyyaml",
        "tqdm",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "qi-align=qi_align.pipeline.pipeline:main_cli",
        ]
    },
    python_requires=">=3.10",
)
