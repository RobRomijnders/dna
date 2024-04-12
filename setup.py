from setuptools import setup, find_packages

setup(
  name='dpgnn',
  version='0.1',
  description='',
  url='',
  author='Rob Romijnders',
  author_email='romijnders@gmail.com',
  license='LICENSE.txt',
  install_requires=[
    'matplotlib',
    'numpy',
    'nose2',
    'pylint',
    'pycodestyle',
    'pytype',
    'pydocstyle',
    'scipy',
    'scikit-learn',
    'wandb',
  ],
  packages=find_packages(exclude=('tests')),
  zip_safe=False)
