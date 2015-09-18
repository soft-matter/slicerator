from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import range
import collections
import itertools
from functools import wraps


class Slicerator(object):

    def __init__(self, ancestor, method, indices=None, length=None,
                 expose_attrs=None):
        """A generator that supports fancy indexing

        When sliced using any iterable with a known length, it return another
        object like itself, a Slicerator. When sliced with an integer,
        it returns the data payload.

        Also, this retains the attributes of the ultimate ancestor that
        created it (or its parent, or its parent's parent, ...).

        Parameters
        ----------
        ancestor : object
        method_name : string
            method of ancestor object that accept an integer as its argument
        indices : iterable
            giving indices into `ancestor`
        length : integer, optional
            length of indicies
            This is required if `indices` is a generator,
            that is, if `len(indices)` is invalid
        expose_attrs : list
            list of attributes to be propaged into Slicerator

        Examples
        --------
        # Slicing on a Slicerator returns another Slicerator...
        >>> v = Slicerator([0, 1, 2, 3], range(4), 4)
        >>> v1 = v[:2]
        >>> type(v[:2])
        Slicerator
        >>> v2 = v[::2]
        >>> type(v2)
        Slicerator
        >>> v2[0]
        0
        # ...unless the slice itself has an unknown length, which makes
        # slicing impossible.
        >>> v3 = v2((i for i in [0]))  # argument is a generator
        >>> type(v3)
        generator
        """
        if indices is None:
            try:
                indices = range(len(ancestor))
            except TypeError:
                raise ValueError("The indices parameter is required in this "
                                 "case because len(ancestor) is not valid.")
        if length is None:
            try:
                length = len(indices)
            except TypeError:
                raise ValueError("The length parameter is required in this "
                                 "case because len(indices) is not valid.")
        if expose_attrs is None:
            expose_attrs = []
        self._len = length
        self._ancestor = ancestor
        self._method = method
        self._indices = indices
        self._attrs = expose_attrs
        self._counter = 0
        self._proc_func = lambda image: image

    @classmethod
    def from_func(cls, func, length, expose_attrs):
        """
        Make a Slicerator from a function that accepts an integer index

        Parameters
        ----------
        func : callback
            callable that accepts an integer as its argument
        length : int
            number of elements; used to supposed revserse slicing like [-1]
        expose_attrs : list
            list of attributes to be propaged into Slicerator
        """
        class Dummy:

            def dummy_method(self):
                return func

            def __len__(self):
                return length

        return cls(Dummy(), 'dummy_method', expose_attrs=expose_attrs)

    @classmethod
    def from_list(cls, obj, expose_attrs=None):
        """
        Construct a Slicerator from an indexable object (like a list).

        This is a convenience function. `Slicereator.from_list(obj)` is
        equivalent to `Slicerator(obj, '__getitem__')`.

        Parameters
        ----------
        obj : list-like
            must support __getitem__ and __len__
        expose_attrs : list, optional
            list of attributes to be propaged into Slicerator
        """
        return cls(obj, '__getitem__', expose_attrs=expose_attrs)

    @property
    def indices(self):
        # Advancing indices won't affect this new copy of self._indices.
        indices, self._indices = itertools.tee(iter(self._indices))
        return indices

    def _get(self, key):
        "Wrap ancestor's method in a processing function."
        return self._proc_func(getattr(self._ancestor, self._method)(key))

    def __repr__(self):
        msg = "Sliced and/or processed {0}. Original repr:\n".format(
                type(self._ancestor).__name__)
        old = '\n'.join("    " + ln for ln in repr(self._ancestor).split('\n'))
        return msg + old

    def __iter__(self):
        return (self._get(i) for i in self.indices)

    def __len__(self):
        return self._len

    def __getattr__(self, key):
        # Remember this only gets called if __getattribute__ raises an
        # AttributeError. Try the ancestor object.
        if key not in self._attrs:
            raise AttributeError
        attr = getattr(self._ancestor, key)
        if isinstance(attr, Slicerator):
            return Slicerator(attr, '__getitem__', self.indices, len(self),
                              self._attrs)
        else:
            return attr

    def __getitem__(self, key):
        """for data access"""
        _len = len(self)
        abs_indices = self.indices

        if isinstance(key, slice):
            # if input is a slice, return another Slicerator
            start, stop, step = key.indices(_len)
            rel_indices = range(start, stop, step)
            new_length = len(rel_indices)
            indices = _index_generator(rel_indices, abs_indices)
            return Slicerator(self._ancestor, '__getitem__', indices, new_length,
                              self._attrs)
        elif isinstance(key, collections.Iterable):
            # if the input is an iterable, doing 'fancy' indexing
            if hasattr(key, '__array__') and hasattr(key, 'dtype'):
                if key.dtype == bool:
                    # if we have a bool array, set up masking but defer
                    # the actual computation, returning another Slicerator
                    nums = range(len(self))
                    # This next line fakes up numpy's bool masking without
                    # importing numpy.
                    rel_indices = [x for x, y in zip(nums, key) if y]
                    indices = _index_generator(rel_indices, abs_indices)
                    new_length = sum(key)
                    return Slicerator(self._ancestor, '__getitem__', indices,
                                      new_length, self._attrs)
            try:
                new_length = len(key)
            except TypeError:
                # The key is a generator; return a plain old generator.
                # Without knowing the length of the *key*,
                # we can't give a Slicerator
                # Also it cannot be checked if values are in range.
                gen = (self[_k if _k >= 0 else _len + _k] for _k in key)
                return gen
            else:
                # The key is a list of in-range values. Build another
                # Slicerator, again deferring computation.
                if any(_k < -_len or _k >= _len for _k in key):
                    raise IndexError("Keys out of range")
                rel_indices = ((_k if _k >= 0 else _len + _k) for _k in key)
                indices = _index_generator(rel_indices, abs_indices)
                return Slicerator(self._ancestor, '__getitem__', indices,
                                  new_length, self._attrs)
        else:
            if key < -_len or key >= _len:
                raise IndexError("Key out of range")
            try:
                abs_key = self._indices[key]
            except TypeError:
                key = key if key >= 0 else _len + key
                for _, i in zip(range(key + 1), self.indices):
                    abs_key = i
            return self._get(abs_key)

    def __getstate__(self):
        # When serializing, return a list of the sliced and processed data
        # Any exposed attrs are lost.
        return [self._get(key) for key in self.indices]

    def __setstate__(self, data_as_list):
        # When deserializing, restore the Slicerator
        return self.__init__(data_as_list, '__getitem__')

    def close(self):
        "Closing this child slice of the original reader does nothing."
        pass


def _index_generator(new_indices, old_indices):
    """Find locations of new_indicies in the ref. frame of the old_indices.
    
    Example: (1, 3), (1, 3, 5, 10) -> (3, 10)

    The point of all this trouble is that this is done lazily, returning
    a generator without actually looping through the inputs."""
    # Use iter() to be safe. On a generator, this returns an identical ref.
    new_indices = iter(new_indices)
    n = next(new_indices)
    last_n = None
    done = False
    while True:
        old_indices_, old_indices = itertools.tee(iter(old_indices))
        for i, o in enumerate(old_indices_):
            # If new_indices is not strictly monotonically increasing, break
            # and start again from the beginning of old_indices.
            if last_n is not None and n <= last_n:
                last_n = None
                break
            if done:
                raise StopIteration
            if i == n:
                last_n = n
                try:
                    n = next(new_indices)
                except StopIteration:
                    done = True
                    # Don't stop yet; we still have one last thing to yield.
                yield o
            else:
                continue


def pipeline(func):
    """Decorator to make function aware of Slicerator objects.

    When the function is applied to a Slicerator, it
    returns another lazily-evaluated, Slicerator object.

    When the function is applied to any other object, it falls back on its
    normal behavhior.

    Parameters
    ----------
    func : callable
        function that accepts an image as its first argument

    Returns
    -------
    processed_images : Slicerator

    Example
    -------
    Apply the pipeline decorator to your image processing function.
    >>> @pipeline
    ...  def color_channel(image, channel):
    ...      return image[channel, :, :]
    ...

    Passing a Slicerator the function returns another Slicerator
    that "lazily" applies the function when the images come out. Different
    functions can be applied to the same underlying images, creating
    independent objects.
    >>> red_images = color_channel(images, 0)
    >>> green_images = color_channel(images, 1)

    Pipeline functions can also be composed.
    >>> @pipeline
    ... def rescale(image):
    ... return (image - image.min())/image.ptp()
    ...
    >>> rescale(color_channel(images, 0))

    The function can still be applied to ordinary images. The decorator
    only takes affect when a Slicerator object is passed.
    >>> single_img = images[0]
    >>> red_img = red_channel(single_img)  # normal behavior
    """
    @wraps(func)
    def process(obj, *args, **kwargs):
        if isinstance(obj, Slicerator):
            s = Slicerator.from_list(obj)
            def f(x):
                return func(x, *args, **kwargs)
            s._proc_func = f
            return s
        else:
            # Fall back on normal behavior of func, interpreting input
            # as a single image.
            return func(obj, *args, **kwargs)

    if process.__doc__ is None:
        process.__doc__ = ''
    process.__doc__ = ("This function has been made lazy. When passed\n"
                       "a Slicerator, it will return a \n"
                       "new Slicerator of the results. When passed \n"
                       "any other objects, its behavior is "
                       "unchanged.\n\n") + process.__doc__
    return process
