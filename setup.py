import os
import versioneer
from setuptools import setup

try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except IOError:
    descr = ''

try:
    from pypandoc import convert
    descr = convert(descr, 'rst', format='md')
except ImportError:
    pass

setup(name='slicerator',
      version=versioneer.get_version(),
      author='Daniel B. Allan',
      author_email='daniel.b.allan@gmail.com',
      py_modules=['slicerator', '_slicerator_version'],
      description='A lazy-loading, fancy-sliceable iterable.',
      url='http://github.com/soft-matter/slicerator',
      install_requires=['six'],
      cmdclass=versioneer.get_cmdclass(),
      platforms='Cross platform (Linux, Mac OSX, Windows)',
      license="BSD",
      classifiers=['Development Status :: 4 - Beta',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   ],
      long_description=descr,
      )
