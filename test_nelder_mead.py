"""
Test module for the Nelder-Mead implementation.
"""
import inspect
from importlib.machinery import SourceFileLoader
import importlib.util
import pytest
from pylint.lint import Run
from pylint.reporters import CollectingReporter
import numpy as np
import nelder_mead as nm


epsilon = 0.0000055


main_module_spec = importlib.util.spec_from_loader("main", SourceFileLoader("main", "main.py"))
main_module = importlib.util.module_from_spec(main_module_spec)


# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def linter_nm():
    """ Test codestyle for src file of nelder_mead """
    src_file = inspect.getfile(nm)
    rep = CollectingReporter()
    # disabled warnings:
    # 0301 line too long
    # 0103 variables name (does not like shorter than 2 chars)
    r = Run(['--disable=C0301,C0103', '-sn', src_file], reporter=rep, exit=False)
    return r.linter


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("limit", range(0, 11))
def test_codestyle_score_nm(linter_nm, limit, runs=None):
    """ Evaluate codestyle for nelder_mead.py """
    if runs is None:
        runs = []
    if len(runs) == 0:
        print('\nLinter output:')
        for m in linter_nm.reporter.messages:
            print(f'{m.msg_id} ({m.symbol}) line {m.line}: {m.msg}')
    runs.append(limit)
    score = linter_nm.stats.global_note

    print(f'pylint score = {score} limit = {limit}')
    assert score >= limit


# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def linter_main():
    """ Test codestyle for src file of main """
    src_file = inspect.getfile(main_module)
    rep = CollectingReporter()
    r = Run(['--disable=C0301,C0103', '-sn', src_file], reporter=rep, exit=False)
    return r.linter


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("limit", range(0, 11))
def test_codestyle_score_main(linter_main, limit, runs=None):
    """ Evaluate codestyle for main.py """
    if runs is None:
        runs = []
    if len(runs) == 0:
        print('\nLinter output:')
        for m in linter_main.reporter.messages:
            print(f'{m.msg_id} ({m.symbol}) line {m.line}: {m.msg}')
    runs.append(limit)
    score = linter_main.stats.global_note

    print(f'pylint score = {score} limit = {limit}')
    assert score >= limit


# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def linter_self():
    """ Test codestyle for current test file """
    src_file = __file__
    rep = CollectingReporter()
    # disabled warnings:
    # 0301 line too long
    # 0103 variables name (does not like shorter than 2 chars)
    # 0913 redefining name '...' from outer scope
    # 0621 too many arguments (x/5)
    r = Run(['--disable=C0301,C0103, R0913, W0621, ', '-sn', src_file], reporter=rep, exit=False)
    return r.linter


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("limit", range(0, 11))
def test_codestyle_score_test(linter_self, limit, runs=None):
    """ Evaluate codestyle for current test file """
    if runs is None:
        runs = []
    if len(runs) == 0:
        print('\nLinter output:')
        for m in linter_self.reporter.messages:
            print(f'{m.msg_id} ({m.symbol}) line {m.line}: {m.msg}')
    runs.append(limit)
    score = linter_self.stats.global_note

    print(f'pylint score = {score} limit = {limit}')
    assert score >= limit


# ----------------------------------------------------------------------------------------------------------------------
def test_zero_point():
    """
    Testing the zero_point(n) function. In this test we check that the function returns a vector of the correct size,
    all elements of which are equal to zero.
    """
    z = nm.zero_point(5)
    assert len(z.vector) == 5
    assert all(v == 0.0 for v in z.vector)


# ----------------------------------------------------------------------------------------------------------------------
def test_start_simplex():
    """
    Testing the StartSimplex(Edge, StartPoint, N) function. We check that the function returns the correct number of
    vertices and that the starting point remains the same.
    """
    start_point = [1, 2, 3]
    edge = 2
    n = 3
    vertices = nm.start_simplex(edge, start_point, n)
    assert len(vertices) == n + 1
    assert vertices[0].vector == start_point


# ----------------------------------------------------------------------------------------------------------------------
def no_extrema2d(x):
    """
    f(x) = x function
    """
    return x[0]


# ----------------------------------------------------------------------------------------------------------------------
def no_extrema3d(x):
    """
    f(x,y) = x + y function
    """
    return x[0] + x[1]


# ----------------------------------------------------------------------------------------------------------------------
def no_extrema4d(x):
    """
    f(x,y,z) = x + y + z function
    """
    return x[0] + x[1] + x[2]


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("function, dimensions, initial_point, scale, tol, expected", [
    (nm.parabola_0, 1, [-1.9], 1, epsilon, [0]),
    (nm.parabola_1, 1, [2], 1, epsilon, [1]),
    (nm.parabola_2, 1, [2], 1, epsilon, [-3]),
    (nm.ff_dim_2, 2, [0, 0], 1, epsilon, [1, 0.5]),
    (nm.parabaloid, 2, [3, 5], 1, epsilon, [0, 0]),
    (nm.rosenbrock, 2, [3, 6], 1, epsilon, [1, 1]),
    (nm.easy_3_dim, 3, [1, 2, 8], 1, epsilon, [0, 0, 0]),
    (nm.harder_3_dim, 3, [1, 1, 1], 1, epsilon, [np.sqrt(2), np.sqrt(2), np.sqrt(2)])
])
def test_neldermead(function, dimensions, initial_point, scale, tol, expected):
    """
    Test the Nelder-Mead implementation with various functions, dimensions, initial points, and expected results.
    """
    result = nm.nelder_mead(function, dimensions, initial_point, scale, tol)
    np.testing.assert_allclose(result[0].get_x(), expected, rtol=tol, atol=tol)


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("function, quant_var, initial_point, scale, tol", [
    (no_extrema2d, 1, [1], 2, epsilon),  # no exrema case check
    (no_extrema3d, 2, [1, 1], 2, epsilon),
    (no_extrema3d, 3, [1, 1, 1], 2, epsilon)
])
def test_neldermead_no_extrema(function, quant_var, initial_point, scale, tol):
    """
    Test the Nelder-Mead implementation for functions with no extrema.
    """
    result = nm.nelder_mead(function, quant_var, initial_point, scale, tol)
    assert result == -1
# ----------------------------------------------------------------------------------------------------------------------
