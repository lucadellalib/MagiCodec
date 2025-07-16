import warnings
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="magicodec",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
)

warnings.warn(
    "MagiCodec requires flash-attn==2.8.1, which needs to be installed manually "
    "(run `bash install.sh` from https://github.com/lucadellalib/MagiCodec)"
)
