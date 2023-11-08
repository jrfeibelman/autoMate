from setuptools import setup, find_packages

setup(name='autoMate',
      version='0.0.1',
      packages=find_packages(include=['autoMate', 'autoMate.*']),
      python_requires=">=3.11,<3.12",
      py_modules=[])