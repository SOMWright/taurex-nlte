from setuptools import find_packages
from distutils.core import setup

packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex_nlte', ]

requires = []

install_requires = ['taurex', ]

entry_points = {'taurex.plugins': 'TauREx Non-LTE = taurex_nlte'}

setup(name='taurex_nlte',
      author="Sam Wright",
      author_email="samuel.wright.13@ucl.ac.uk",
      license="BSD",
      description='A Taurex3 plugin to implement non-LTE functionality based on bi-temperature approximated cross sections',
      packages=packages,
      entry_points=entry_points,
      provides=provides,
      requires=requires,
      install_requires=install_requires,
      version="0.0.1")