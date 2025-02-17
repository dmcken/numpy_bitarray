'''Basic tests'''

# System import
import random

# Local import
import numpy_bitarray

def test_create_small():
    """Create a small bitarray, set a value and confirm its set.
    """
    x = numpy_bitarray.Bitarray(40)

    x[5] = 1

    assert x[5] == 1

def test_random_create_medium():
    """Create a medium sized bitarray.
    """
    size = random.randint(2**16, 2**24)
    x = numpy_bitarray.Bitarray(size)

    # set a few items and then test all of them
    checks = random.randint(10,100)
    nums = set()
    for _ in range(checks):
        to_set = random.randint(0,size)
        nums.add(to_set)
        x[to_set] = 1

        if x[to_set] != 1:
            raise AssertionError(f"Error, value not found: {to_set}")

    assert x.bitcount() == len(nums)
