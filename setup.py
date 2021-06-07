  
from setuptools import setup, find_namespace_packages


setup(
    name = 'aspectator',
    version = '0.1',
    description = '',
    packages = find_namespace_packages(where = 'aspectator'),
    package_dir = {'': 'aspectator'},
    include_package_data = True,
    install_requires = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'nltk',
        'spacy',
        'sklearn',
        'scikit-learn-extra'
    ],
    python_requires = '>=3.8'
)
