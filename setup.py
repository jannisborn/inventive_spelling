"""Install package."""
from setuptools import setup, find_packages
setup(
    name='invspell',
    version='0.0.1',
    description=(
        'Implementation of computational model for inventive spelling'
    ),
    long_description=open('README.md').read(),
    url='https://github.com/jannisborn/inventive_spelling',
    author='Jannis Born',
    author_email='jannis.born@gmx.de',
    install_requires=[
        'numpy', 'tensorflow', 'scikit-learn', 'matplotlib'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)