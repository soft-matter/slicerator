Slicerator
==========

a lazy-loading, fancy-slicable iterable

Think of it like a generator that is "reusable" and has a length.

[![build status](https://travis-ci.org/soft-matter/slicerator.png?branch=master)](https://travis-ci.org/soft-matter/slicerator)

Installation
------------

On any platform, use pip or conda.

`pip install slicerator`

or

`conda install -c soft-matter slicerator`

Example
-------

```
from slicerator import Slicerator

class MyLazyLoader:

    def __getitem__(self, i):
        # If a specific item is requested, load it and return it.
        # Otherwise, return a lazy-loading Slicerator.
        if isinstance(i, int):
            # load thing number i
            return thing
        else:
            return Slicerator(self, range(len(self)), len(self))[i]

    def __len__(self):
        # do stuff
        return number_of_things

    def __iter__(self):
        return iter(self[:])

# Demo:
>>> a = MyLazyLoader()
>>> s1 = a[::2]  # no data is loaded yet
>>> s2 = s1[1:]  # no data is loaded yet
>>> some_data = s2[0]
```
