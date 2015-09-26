from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import random
import types
import nose
from six import BytesIO
import pickle
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from slicerator import Slicerator, pipeline

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def assert_letters_equal(actual, expected):
    # check if both lengths are equal
    assert_equal(len(actual), len(expected))
    for actual_, expected_ in zip(actual, expected):
        assert_equal(actual_, expected_)


def compare_slice_to_list(actual, expected):
    assert_letters_equal(actual, expected)
    indices = list(range(len(actual)))
    for i in indices:
        # test positive indexing
        assert_letters_equal(actual[i], expected[i])
        # test negative indexing
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # in reverse order
    for i in indices[::-1]:
        assert_letters_equal(actual[i], expected[i])
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # in shuffled order (using a consistent random seed)
    r = random.Random(5)
    r.shuffle(indices)
    for i in indices:
        assert_letters_equal(actual[i], expected[i])
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # test list indexing
    some_indices = [r.choice(indices) for _ in range(2)]
    assert_letters_equal([actual[i] for i in some_indices],
                         [expected[i] for i in some_indices])
    # mixing positive and negative indices
    some_indices = [r.choice(indices + [-i-1 for i in indices])
                    for _ in range(2)]
    assert_letters_equal([actual[i] for i in some_indices],
                         [expected[i] for i in some_indices])
    # test slices
    assert_letters_equal(actual[::2], expected[::2])
    assert_letters_equal(actual[1::2], expected[1::2])
    assert_letters_equal(actual[::3], expected[::3])
    assert_letters_equal(actual[1:], expected[1:])
    assert_letters_equal(actual[:], expected[:])
    assert_letters_equal(actual[:-1], expected[:-1])


v = Slicerator(list('abcdefghij'))


def test_bool_mask():
    mask = np.array([True, False] * 5)
    s = v[mask]
    assert_letters_equal(s, list('acegi'))


def test_slice_of_slice():
    slice1 = v[4:]
    compare_slice_to_list(slice1, list('efghij'))
    slice2 = slice1[-3:]
    compare_slice_to_list(slice2, list('hij'))
    slice1a = v[[3, 4, 5, 6, 7, 8, 9]]
    compare_slice_to_list(slice1a, list('defghij'))
    slice2a = slice1a[::2]
    compare_slice_to_list(slice2a, list('dfhj'))
    slice2b = slice1a[::-1]
    compare_slice_to_list(slice2b, list('jihgfed'))
    slice2c = slice1a[::-2]
    compare_slice_to_list(slice2c, list('jhfd'))
    slice2d = slice1a[:0:-1]
    compare_slice_to_list(slice2d, list('jihgfe'))
    slice2e = slice1a[-1:1:-1]
    compare_slice_to_list(slice2e, list('jihgf'))
    slice2f = slice1a[-2:1:-1]
    compare_slice_to_list(slice2f, list('ihgf'))
    slice2g = slice1a[::-3]
    compare_slice_to_list(slice2g, list('jgd'))
    slice2h = slice1a[[5, 6, 2, -1, 3, 3, 3, 0]]
    compare_slice_to_list(slice2h, list('ijfjgggd'))


def test_slice_of_slice_of_slice():
    slice1 = v[4:]
    compare_slice_to_list(slice1, list('efghij'))
    slice2 = slice1[1:-1]
    compare_slice_to_list(slice2, list('fghi'))
    slice2a = slice1[[2, 3, 4]]
    compare_slice_to_list(slice2a, list('ghi'))
    slice3 = slice2[1::2]
    compare_slice_to_list(slice3, list('gi'))


def test_slice_of_slice_of_slice_of_slice():
    # Take the red pill. It's slices all the way down!
    slice1 = v[4:]
    compare_slice_to_list(slice1, list('efghij'))
    slice2 = slice1[1:-1]
    compare_slice_to_list(slice2, list('fghi'))
    slice3 = slice2[1:]
    compare_slice_to_list(slice3, list('ghi'))
    slice4 = slice3[1:]
    compare_slice_to_list(slice4, list('hi'))

    # Give me another!
    slice1 = v[2:]
    compare_slice_to_list(slice1, list('cdefghij'))
    slice2 = slice1[0::2]
    compare_slice_to_list(slice2, list('cegi'))
    slice3 = slice2[:]
    compare_slice_to_list(slice3, list('cegi'))
    print('define slice4')
    slice4 = slice3[:-1]
    print('compare slice4')
    compare_slice_to_list(slice4, list('ceg'))
    print('define slice4a')
    slice4a = slice3[::-1]
    print('compare slice4a')
    compare_slice_to_list(slice4a, list('igec'))


def test_slice_with_generator():
    slice1 = v[1:]
    compare_slice_to_list(slice1, list('bcdefghij'))
    slice2 = slice1[(i for i in range(2, 5))]
    assert_letters_equal(list(slice2), list('def'))
    assert_true(isinstance(slice2, types.GeneratorType))


def test_no_len_raises():
    with assert_raises(ValueError):
        Slicerator((i for i in range(5)), (i for i in range(5)))


def _capitalize(letter):
    return letter.upper()


def _capitalize_if_equal(letter, other_letter):
    if letter == other_letter:
        return letter.upper()
    else:
        return letter


def _a_to_z(letter):
    if letter == 'a':
        return 'z'
    else:
        return letter


def test_pipeline_simple():
    capitalize = pipeline()(_capitalize)
    cap_v = capitalize(v[:1])

    assert_letters_equal([cap_v[0]], [_capitalize(v[0])])


def test_repr():
    repr(v)


def test_getattr():
    class MyList(list):
        attr1 = 'hello'
        attr2 = 'hello again'

        def s(self, i):
            return list('ABCDEFGHIJ')[i]

        def close(self):
            pass

    a = Slicerator(MyList('abcdefghij'), propagate=['attr1'],
                   propagate_indexed=['s'])
    assert_letters_equal(a, list('abcdefghij'))
    assert_true(hasattr(a, 'attr1'))
    assert_false(hasattr(a, 'attr2'))
    assert_true(hasattr(a, 's'))
    assert_false(hasattr(a, 'close'))
    assert_equal(a.attr1, 'hello')
    with assert_raises(AttributeError):
        a[:5].nonexistent_attr

    s1 = a[::2].s
    assert_equal([s1(i) for i in range(5)], list('ACEGI'))
    s2 = a[::2][1:].s
    assert_equal([s2(i) for i in range(4)], list('CEGI'))
    assert_equal(a[::2][1:].s(0), 'C')


def test_pipeline_with_args():
    capitalize = pipeline()(_capitalize_if_equal)
    cap_a = capitalize(v, 'a')
    cap_b = capitalize(v, 'b')

    assert_letters_equal(cap_a, 'Abcdefghij')
    assert_letters_equal(cap_b, 'aBcdefghij')
    assert_letters_equal([cap_a[0]], ['A'])
    assert_letters_equal([cap_b[0]], ['a'])
    assert_letters_equal([cap_a[0]], ['A'])


def test_composed_pipelines():
    a_to_z = pipeline()(_a_to_z)
    capitalize = pipeline()(_capitalize_if_equal)

    composed = capitalize(a_to_z(v), 'c')

    assert_letters_equal(composed, 'zbCdefghij')

def test_serialize():
    # dump Slicerator
    stream = BytesIO()
    pickle.dump(v, stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('abcdefghij'))
    compare_slice_to_list(v2[4:], list('efghij'))
    compare_slice_to_list(v2[4:][:-1], list('efghi'))

    # dump sliced Slicerator
    stream = BytesIO()
    pickle.dump(v[4:], stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('efghij'))
    compare_slice_to_list(v2[2:], list('ghij'))
    compare_slice_to_list(v2[2:][:-1], list('ghi'))

    # dump sliced sliced Slicerator
    stream = BytesIO()
    pickle.dump(v[4:][:-1], stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('efghi'))
    compare_slice_to_list(v2[2:], list('ghi'))
    compare_slice_to_list(v2[2:][:-1], list('gh'))

    # test pipeline
    capitalize = pipeline()(_capitalize_if_equal)
    stream = BytesIO()
    pickle.dump(capitalize(v, 'a'), stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('Abcdefghij'))

def test_class():
    @Slicerator.from_class
    class Dummy(object):
        propagate_indexed = ['time', 'return_i']
        propagate = ['filename', 'other_attr']
        def __init__(self):
            self.frame = list('abcdefghij')

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            return self.frame[i]  # actual code of get_frame

        def time(self, i):
            return i * 5

        def return_i(self, i):
            return i

        @property
        def filename(self):
            return 'filename'

        def other_attr(self):
            return 'other_string'

    dummy = Dummy()
    compare_slice_to_list(dummy, 'abcdefghij')
    compare_slice_to_list(dummy[1:], 'bcdefghij')
    assert_equal(dummy[1:].return_i(0), 1)
    assert_equal(dummy[1:].time(0), 5)
    assert_equal(dummy[1:].filename, 'filename')
    assert_equal(dummy[1:].other_attr(), 'other_string')

    compare_slice_to_list(dummy[1:][2:], 'defghij')
    assert_equal(dummy[1:][2:].return_i(0), 3)
    assert_equal(dummy[1:][2:].time(0), 15)
    assert_equal(dummy[1:][2:].filename, 'filename')
    assert_equal(dummy[1:][2:].other_attr(), 'other_string')

    capitalize = pipeline()(_capitalize_if_equal)
    cap_b = capitalize(dummy, 'b')
    assert_letters_equal(cap_b, 'aBcdefghij')

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
