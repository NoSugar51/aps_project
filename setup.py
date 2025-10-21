from setuptools import setup, find_packages

setup(
    name="aps-project",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    python_requires=">=3.10",
)
