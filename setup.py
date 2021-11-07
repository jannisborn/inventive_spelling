"""Install package."""
from setuptools import setup, find_packages
setup(
    name='invspell',
    version='0.0.1',
    description=(
        'Implementation of computational model for inventive spelling'
    ),
    long_description=open('README.md').read(),
    install_requires=[
        'numpy', 'tensorflow', 'scikit-learn', 'matplotlib'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)