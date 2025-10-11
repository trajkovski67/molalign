from setuptools import setup, find_packages

setup(
    name="molalign",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tblite",
    ],
    entry_points={
        "console_scripts": [
            "molalign=molalign.run_molalign_tb:main"
        ],
    },
    python_requires='>=3.7',
)

