from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import range
import collections
import itertools
from functools import wraps


# set version string using versioneer
from _slicerator_version import get_versions
__version__ = get_versions()['version']
del get_versions


def _iter_attr(obj):
    try:
        for ns in [obj] + obj.__class__.mro():
            for attr in ns.__dict__:
                yield ns.__dict__[attr]
    except AttributeError:
        raise StopIteration  # obj has no __dict__


class Slicerator(object):
    def __init__(self, ancestor, indices=None, length=None,
                 propagate_attrs=None, proc_func=None):
        """A generator that supports fancy indexing

        When sliced using any iterable with a known length, it returns another
        object like itself, a Slicerator. When sliced with an integer,
        it returns the data payload.

        Also, the attributes of the parent object can be propagated, exposed
        through the child Slicerators. By default, no attributes are
        propagated. Attributes can be white_listed by using the optional
        parameter `propagated_attrs`.

        Class methods taking an index will be remapped if they are decorated
        with `index_attr`. They also have to be present in the
        `propagate_attrs` list.

        Parameters
        ----------
        ancestor : object
        indices : iterable
            Giving indices into `ancestor`.
            Required if len(ancestor) is invalid.
        length : integer
            length of indices
            This is required if `indices` is a generator,
            that is, if `len(indices)` is invalid
        propagate_attrs : list of str, optional
            list of attributes to be propagated into Slicerator
        proc_func : function
            function that processes data returned by Slicerator. The function
            acts element-wise and is only evaluated when data is actually
            returned

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
        if indices is None and length is None:
            try:
                length = len(ancestor)
                indices = range(length)
            except TypeError:
                raise ValueError("The length parameter is required in this "
                                 "case because len(ancestor) is not valid.")
        elif indices is None:
            indices = range(length)
        elif length is None:
            try:
                length = len(indices)
            except TypeError:
                raise ValueError("The length parameter is required in this "
                                 "case because len(indices) is not valid.")

        # when list of propagated attributes are given explicitly,
        # take this list and ignore the class definition
        if propagate_attrs is not None:
            self._propagate_attrs = propagate_attrs
        else:
            # check propagated_attrs field from the ancestor definition
            if hasattr(ancestor, 'propagate_attrs'):
                self._propagate_attrs = ancestor.propagate_attrs
            else:
                self._propagate_attrs = []

            # add methods having the _propagate hook
            for attr in _iter_attr(ancestor):
                if hasattr(attr, '_propagate_hook'):
                    self._propagate_attrs.append(attr.__name__)

        self._len = length
        self._ancestor = ancestor
        self._indices = indices
        if proc_func is None:
            self._proc_func = lambda x: x
        else:
            self._proc_func = proc_func

    @classmethod
    def from_func(cls, func, length, propagate_attrs=None):
        """
        Make a Slicerator from a function that accepts an integer index

        Parameters
        ----------
        func : callback
            callable that accepts an integer as its argument
        length : int
            number of elements; used to supposed revserse slicing like [-1]
        propagate_attrs : list, optional
            list of attributes to be propagated into Slicerator
        """
        class Dummy:

            def __getitem__(self, i):
                return func(i)

            def __len__(self):
                return length

        return cls(Dummy(), propagate_attrs=propagate_attrs)

    @classmethod
    def from_class(cls, some_class, propagate_attrs=None):
        """Make an existing class support fancy indexing via Slicerator objects.

        When sliced using any iterable with a known length, it returns a
        Slicerator. When sliced with an integer, it returns the data payload.

        Also, the attributes of the parent object can be propagated, exposed
        through the child Slicerators. By default, no attributes are
        propagated. Attributes can be white_listed in the following ways:

        1. using the optional parameter `propagate_attrs`; the contents of this
           list will overwrite any other list of propagated attributes
        2. using the @propagate_attr decorator inside the class definition
        3. using a `propagate_attrs` class attribute inside the class definition

        The difference between options 2 and 3 appears when subclassing. As
        option 2 is bound to the method, the method will always be propagated.
        On the contrary, option 3 is bound to the class, so this can be
        overwritten by the subclass.

        Class methods taking an index will be remapped if they are decorated
        with `index_attr`. This decorator does not ensure that the method is
        propagated.

        The existing class should support indexing (method __getitem__) and
        it should define a length (method __len__).

        The result will look exactly like the existing class (__name__, __doc__,
        __module__, __repr__ will be propagated), but the __getitem__ will be
        renamed to _get and __getitem__ will produce a Slicerator object
        when sliced.

        Parameters
        ----------
        some_class : class
        propagated_attrs : list, optional
            list of attributes to be propagated into Slicerator
            this will overwrite any other propagation list
        """

        class Slicerator_subclass(some_class):
            _slicerator_hook = True
            _get = some_class.__getitem__
            if hasattr(some_class, '__doc__'):
                __doc__ = some_class.__doc__  # for Python 2, do it here

            def __getitem__(self, i):
                """Getitem supports repeated slicing via Slicerator objects."""
                indices, new_length = key_to_indices(i, len(self))
                if new_length is None:
                    return self._get(indices)
                else:
                    return cls(self, indices, new_length, propagate_attrs)

        for name in ['__name__', '__module__', '__repr__']:
            try:
                setattr(Slicerator_subclass, name, getattr(some_class, name))
            except AttributeError:
                pass

        return Slicerator_subclass

    @property
    def indices(self):
        # Advancing indices won't affect this new copy of self._indices.
        indices, self._indices = itertools.tee(iter(self._indices))
        return indices

    def _get(self, key):
        "Wrap ancestor's method in a processing function."
        return self._proc_func(self._ancestor[key])

    def _map_index(self, key):
        if key < -self._len or key >= self._len:
            raise IndexError("Key out of range")
        try:
            abs_key = self._indices[key]
        except TypeError:
            key = key if key >= 0 else self._len + key
            for _, i in zip(range(key + 1), self.indices):
                abs_key = i
        return abs_key

    def __repr__(self):
        msg = "Sliced and/or processed {0}. Original repr:\n".format(
                type(self._ancestor).__name__)
        old = '\n'.join("    " + ln for ln in repr(self._ancestor).split('\n'))
        return msg + old

    def __iter__(self):
        return (self._get(i) for i in self.indices)

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        """for data access"""
        if not (isinstance(key, slice) or
                isinstance(key, collections.Iterable)):
            return self._get(self._map_index(key))
        else:
            rel_indices, new_length = key_to_indices(key, len(self))
            if new_length is None:
                return (self[k] for k in rel_indices)
            indices = _index_generator(rel_indices, self.indices)
            return Slicerator(self._ancestor, indices, new_length,
                              self._propagate_attrs, self._proc_func)

    def __getattr__(self, name):
        # to avoid infinite recursion, always check if public field is there
        if '_propagate_attrs' not in self.__dict__:
            self._propagate_attrs = []
        if name in self._propagate_attrs:
            attr = getattr(self._ancestor, name)
            if hasattr(attr, '_index_hook'):
                return SliceableAttribute(self, getattr(self._ancestor, name))
            else:
                return getattr(self._ancestor, name)
        raise AttributeError

    def __getstate__(self):
        # When serializing, return a list of the sliced and processed data
        # Any exposed attrs are lost.
        return [self._get(key) for key in self.indices]

    def __setstate__(self, data_as_list):
        # When deserializing, restore the Slicerator
        return self.__init__(data_as_list)


def key_to_indices(key, length):
    """Converts a fancy key into a list of indices.

    Parameters
    ----------
    key : slice, iterable of numbers, or boolean mask
    length : integer
        length of object that will be indexed

    Returns
    -------
    indices, new_length
    """
    if isinstance(key, slice):
        # if we have a slice, return a range object returning the indices
        start, stop, step = key.indices(length)
        indices = range(start, stop, step)
        return indices, len(indices)

    if isinstance(key, collections.Iterable):
        # if the input is an iterable, doing 'fancy' indexing
        if hasattr(key, '__array__') and hasattr(key, 'dtype'):
            if key.dtype == bool:
                # if we have a bool array, set up masking and return indices
                nums = range(length)
                # This next line fakes up numpy's bool masking without
                # importing numpy.
                indices = [x for x, y in zip(nums, key) if y]
                return indices, sum(key)
        try:
            new_length = len(key)
        except TypeError:
            # The key is a generator; return a plain old generator.
            # Withoug using the generator, we cannot know its length.
            # Also it cannot be checked if values are in range.
            gen = ((_k if _k >= 0 else length + _k) for _k in key)
            return gen, None
        else:
            # The key is a list of in-range values. Check if they are in range.
            if any(_k < -length or _k >= length for _k in key):
                raise IndexError("Keys out of range")
            rel_indices = ((_k if _k >= 0 else length + _k) for _k in key)
            return rel_indices, new_length

    # other cases: it's possibly a number
    try:
        key = int(key)
    except TypeError:
        pass
    else:
        # allow negative indexing
        if -length < key < 0:
            return length + key, None

    # in all other case, just return the key and let user deal with the type.
    return key, None


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
        if hasattr(obj, '_slicerator_hook'):
            def f(x):
                return func(x, *args, **kwargs)
            return Slicerator(obj, proc_func=f)
        elif isinstance(obj, Slicerator):
            def f(x):
                return func(obj._proc_func(x), *args, **kwargs)
            return Slicerator(obj, proc_func=f)
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


def propagate_attr(func):
    func._propagate_hook = True
    return func


def index_attr(func):
    @wraps(func)
    def wrapper(obj, key, *args, **kwargs):
        indices = key_to_indices(key, len(obj))[0]
        if isinstance(indices, collections.Iterable):
            return (func(obj, i, *args, **kwargs) for i in indices)
        else:
            return func(obj, indices, *args, **kwargs)
    wrapper._index_hook = True
    return wrapper


class SliceableAttribute(object):
    """This class enables index-taking methods that are linked to a Slicerator
    object to remap their indices according to the Slicerator indices.

    It also enables fancy indexing, exactly like the Slicerator itself. The new
    attribute supports both calling and indexing to give identical results."""

    def __init__(self, slicerator, attribute):
        self._ancestor = slicerator._ancestor
        self._len = slicerator._len
        self._get = attribute
        self._indices = slicerator._indices

    @property
    def indices(self):
        # Advancing indices won't affect this new copy of self._indices.
        indices, self._indices = itertools.tee(iter(self._indices))
        return indices

    def _map_index(self, key):
        if key < -self._len or key >= self._len:
            raise IndexError("Key out of range")
        try:
            abs_key = self._indices[key]
        except TypeError:
            key = key if key >= 0 else self._len + key
            for _, i in zip(range(key + 1), self.indices):
                abs_key = i
        return abs_key

    def __iter__(self):
        return (self._get(i) for i in self.indices)

    def __len__(self):
        return self._len

    def __call__(self, key, *args, **kwargs):
        if not (isinstance(key, slice) or
                isinstance(key, collections.Iterable)):
            return self._get(self._map_index(key), *args, **kwargs)
        else:
            rel_indices, new_length = key_to_indices(key, len(self))
            return (self[k] for k in rel_indices)

    def __getitem__(self, key):
        return self(key)
