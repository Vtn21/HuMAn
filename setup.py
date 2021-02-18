import setuptools

REQUIREMENTS = ["numpy", "tensorflow", "tqdm", "trimesh"]

setuptools.setup(
    name='human',
    version='0.0.1',
    author="Victor T. N.",
    install_requires=REQUIREMENTS,
    description="HuMAn: Human Motion Anticipation",
    url="https://github.com/Vtn21/HuMAn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
