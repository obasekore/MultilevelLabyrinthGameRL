import setuptools

from pathlib import Path

setuptools.setup(
    name='labyrinth_ctrl',
    version = '0.0.2',
    description = 'An OpenAI Env for Ball In Maze Game',
    long_description = Path("README.md").read_text(),
    long_description_content_type = 'text/markdown',
    packages = setuptools.find_packages(include='labyrinth_ctrl'),
    install_requires=['gym','numpy','matplotlib','pyrep']
)
