from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='genml',
    version='0.4.0',
    description='A Python package for generating Mittag-Leffler correlated noise',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16.0',
        'tqdm>=4.29.1',
        'scipy>=1.2.0',
        'matplotlib>=3.0.2',
        'multiprocess>=0.70.12'
    ],
    python_requires='>=3.6',
)
