import sys
from os.path import dirname as d
from os.path import abspath


root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)


# helper function
def is_almost_equal(number_a, number_b, digit_tolerance):
    eps = 10 ** -digit_tolerance
    if number_b == 0:
        return (number_b == number_a)
    return abs(number_a / number_b - 1) < eps
