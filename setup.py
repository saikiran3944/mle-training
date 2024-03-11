from setuptools import setup

setup(
    name="housing",
    version="0.1",
    description="Modelling housing data.",
    url="https://github.com/saikiran3944/mle-training",
    author="Saikiran",
    author_email="saikiran.badugu@tigeranalytics.com",
    packages=["src\housing"],
    python_requires=">=3.7, <4",
    zip_safe=False,
    extras_require={
        "test": ["pytest", "coverage"],
    },
)
