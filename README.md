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

@Slicerator.from_class
class MyLazyLoader:
    def __getitem__(self, i):
        # this method will be wrapped by Slicerator, so that it accepts slices,
        # lists of integers, or boolean masks. Code below will only be executed
        # when an integer is used.

        # load thing number i
        return thing

    def __len__(self):
        # do stuff
        return number_of_things


# Demo:
>>> a = MyLazyLoader()
>>> s1 = a[::2]  # no data is loaded yet
>>> s2 = s1[1:]  # no data is loaded yet
>>> some_data = s2[0]
```
