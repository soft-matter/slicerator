from setuptools import setup

setup(name='history',
      version='0.1.0',
      author='Daniel B. Allan'
      py_modules=['sliceable_iterable'],
      description='A lazy-loading, fancy-sliceable iterable.',
      url='http://github.com/soft-matter/sliceable-iterable',
      platforms='Cross platform (Linux, Mac OSX, Windows)',
      requires=['six']
      )
