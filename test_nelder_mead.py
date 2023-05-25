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
import simplex as smpx
import point as pt


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


# ======================================================================================================================
@pytest.mark.parametrize("vector, n", [
    ([1, 2, 3], 3),
    ([0, 0, 0, 0], 4),
    ([-1, -2], 2)
])
def test_point_creation(vector, n):
    """
    Test case for creating a Point object.
    It checks if the vector and dimension of the created Point object match the expected values.

    :param vector: The vector representing the coordinates of the point.
    :param n: The dimension of the point.
    """
    point = pt.Point(vector, n)
    assert point.vector == vector
    assert point.n == n


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("point1, point2, expected", [
    (pt.Point([1, 2], 2), pt.Point([3, 4], 2), pt.Point([4, 6], 2)),
    (pt.Point([0, 0], 2), pt.Point([-1, -1], 2), pt.Point([-1, -1], 2)),
    (pt.Point([1, 2, 3], 3), pt.Point([0, 0, 0], 3), pt.Point([1, 2, 3], 3))
])
def test_point_addition(point1, point2, expected):
    """
    Test case for adding two Point objects.
    It checks if the addition of two points produces the expected result.

    :param point1: The first Point object.
    :param point2: The second Point object.
    :param expected: The expected result of the addition.
    """
    result = point1 + point2
    assert result.vector == expected.vector
    assert result.n == expected.n


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("point1, point2, expected", [
    (pt.Point([1, 2], 2), pt.Point([3, 4], 2), pt.Point([-2, -2], 2)),
    (pt.Point([0, 0], 2), pt.Point([-1, -1], 2), pt.Point([1, 1], 2)),
    (pt.Point([1, 2, 3], 3), pt.Point([0, 0, 0], 3), pt.Point([1, 2, 3], 3))
])
def test_point_subtraction(point1, point2, expected):
    """
    Test case for subtracting two Point objects.
    It checks if the subtraction of two points produces the expected result.

    :param point1: The first Point object.
    :param point2: The second Point object.
    :param expected: The expected result of the subtraction.
    """
    result = point1 - point2
    assert result.vector == expected.vector
    assert result.n == expected.n


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("point, scalar, expected", [
    (pt.Point([1, 2], 2), 2, pt.Point([2, 4], 2)),
    (pt.Point([0, 0], 2), 0.5, pt.Point([0, 0], 2)),
    (pt.Point([1, 2, 3], 3), -1, pt.Point([-1, -2, -3], 3))
])
def test_point_multiplication(point, scalar, expected):
    """
    Test case for multiplying a Point object by a scalar.
    It checks if the multiplication of a point by a scalar produces the expected result.

    :param point: The Point object.
    :param scalar: The scalar value.
    :param expected: The expected result of the multiplication.
    """
    result = point * scalar
    assert result.vector == expected.vector
    assert result.n == expected.n


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("point, scalar, expected", [
    (pt.Point([2, 4], 2), 2, pt.Point([1, 2], 2)),
    (pt.Point([0, 0], 2), 0.5, pt.Point([0, 0], 2)),
    (pt.Point([1, 2, 3], 3), -1, pt.Point([-1, -2, -3], 3))
])
def test_point_division(point, scalar, expected):
    """
    Test case for dividing a Point object by a scalar.
    It checks if the division of a point by a scalar produces the expected result.

    :param point: The Point object.
    :param scalar: The scalar value.
    :param expected: The expected result of the division.
    """
    result = point / scalar
    assert result.vector == expected.vector
    assert result.n == expected.n


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("point, expected", [
    (pt.Point([1, 2, 3], 3), [1, 2, 3]),
    (pt.Point([0, 0, 0], 3), [0, 0, 0]),
    (pt.Point([-1, -2], 2), [-1, -2])
])
def test_point_get_x(point, expected):
    """
    Test case for getting the coordinates of a Point object.
    It checks if the coordinates of the Point object match the expected values.

    :param point: The Point object.
    :param expected: The expected coordinates.
    """
    result = point.get_x()
    assert result == expected


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [
    2,
    3,
    4
])
def test_zero_point_creation(n):
    """
    Test case for creating a zero Point object.
    It checks if the created zero Point object has the expected vector and dimension.

    :param n: The dimension of the zero Point object.
    """
    point = pt.zero_point(n)
    assert point.vector == [0.0] * n
    assert point.n == n


# ======================================================================================================================
def are_points_close(point1, point2, tolerance):
    """
    Helper function to check if two points are close within a given tolerance.

    :param point1: First point object.
    :param point2: Second point object.
    :param tolerance: Tolerance for comparing coordinates.
    :return: True if the points are close; False otherwise.
    """
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(point1.get_x(), point2.get_x()))


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("start_point, n, edge, f, expected_vertices", [
    ([0], 1, 2.0, lambda x: x[0] ** 2, [pt.Point([0], 1), pt.Point([2.0], 1)])
])
def test_simplex_initialization(start_point, n, edge, f, expected_vertices):
    """
    Test case for initializing a Simplex object.
    It checks if the Simplex object is correctly initialized with the expected vertices.

    :param start_point: The initial point coordinates.
    :param n: The dimension - 1.
    :param edge: The initial simplex edge length.
    :param f: The function to be minimized.
    :param expected_vertices: The expected vertices of the Simplex object.
    """
    simplex = smpx.Simplex(start_point, n, edge, f)
    vertices = [simplex.get_vertex(i) for i in range(n + 1)]
    assert all(
        are_points_close(v, ev, tolerance=1e-6)
        for v, ev in zip(vertices, expected_vertices)
    )


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("edge, start_point, n, expected_result", [
    (2.0, [0], 1, [pt.Point([0], 1), pt.Point([2.0], 1)]),
    (1.0, [1, 2], 2, [
        pt.Point([1, 2], 2),
        pt.Point([1.97, 2.26], 2),
        pt.Point([1.26, 2.97, 3.0], 2)
    ])
])
def test_start_simplex(edge, start_point, n, expected_result):
    """
    Test case for creating a simplex.
    It checks if the created simplex has the expected vertices.

    :param edge: The initial simplex edge length.
    :param start_point: The initial point coordinates.
    :param n: The dimension - 1.
    :param expected_result: The expected vertices of the simplex.
    """

    simplex_vertices = smpx.start_simplex(edge, start_point, n)
    print(simplex_vertices)
    assert all(
        are_points_close(sv, ev, tolerance=1e-2)
        for sv, ev in zip(simplex_vertices, expected_result)
    )


# ======================================================================================================================
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
@pytest.mark.parametrize("function, dimensions, initial_point, edge, tol, expected", [
    (nm.parabola_0, 1, [-1.9], 1, epsilon, [0]),
    (nm.parabola_1, 1, [2], 1, epsilon, [1]),
    (nm.parabola_2, 1, [2], 1, epsilon, [-3]),
    (nm.ff_dim_2, 2, [0, 0], 1, epsilon, [1, 0.5]),
    (nm.parabaloid, 2, [3, 5], 1, epsilon, [0, 0]),
    (nm.rosenbrock, 2, [3, 6], 1, epsilon, [1, 1]),
    (nm.easy_3_dim, 3, [1, 2, 8], 1, epsilon, [0, 0, 0]),
    (nm.harder_3_dim, 3, [1, 1, 1], 1, epsilon, [np.sqrt(2), np.sqrt(2), np.sqrt(2)])
])
def test_neldermead(function, dimensions, initial_point, edge, tol, expected):
    """
    Test the Nelder-Mead implementation with various functions, dimensions, initial points, and expected results.

    :param function: The function to be minimized.
    :param dimensions: The dimension - 1.
    :param initial_point: The initial point coordinates.
    :param edge: The simplex edge length.
    :param tol: Tolerance for comparing coordinates.
    :param expected: The expected extrema coordinates.
    """
    result = nm.nelder_mead(function, dimensions, initial_point, edge, tol)
    np.testing.assert_allclose(result[0].get_x(), expected, rtol=tol, atol=tol)


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("function, quant_var, initial_point, edge, tol", [
    (no_extrema2d, 1, [1], 2, epsilon),  # no exrema case check
    (no_extrema3d, 2, [1, 1], 2, epsilon),
    (no_extrema4d, 3, [1, 1, 1], 2, epsilon)
])
def test_neldermead_no_extrema(function, quant_var, initial_point, edge, tol):
    """
    Test the Nelder-Mead implementation for functions with no extrema.

    :param function: The function to be minimized.
    :param quant_var: The dimension - 1.
    :param initial_point: The initial point coordinates.
    :param edge: The simplex edge length.
    :param tol: Tolerance for comparing coordinates.
    """
    result = nm.nelder_mead(function, quant_var, initial_point, edge, tol)
    assert result == -1


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("function, n, initial_point, edge, tol, expected", [
    (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2, 4, [1, 2, 3, 4], 1, 0.00001, [0, 0, 0, 0]),
    (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2, 5, [1, 2, 3, 4, 5], 1, 0.000001, [0, 0, 0, 0, 0]),
    (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2, 6, [1, 2, 3, 4, 5, 6], 1, 0.000001, [0, 0, 0, 0, 0, 0])
])
def test_nelder_mead_high_dimensions(function, n, initial_point, edge, tol, expected):
    """
    Test the Nelder-Mead implementation with functions having a high number of dimensions(5, 6, 7).

    This test verifies the behavior of the Nelder-Mead algorithm when applied to functions with
    a higher number of dimensions/variables. The expected extrema points are known and compared
    with the computed extrema points using a given tolerance.

    :param function: The function to be minimized/maximized.
    :param n: The number of variables of the function.
    :param initial_point: The initial point for the algorithm.
    :param edge: The initial simplex edge size scaling factor.
    :param tol: The tolerance for comparing the expected and computed extrema.
    :param expected: The expected extrema point.
    :return: None
    """
    result = nm.nelder_mead(function, n, initial_point, edge, tol)
    np.testing.assert_allclose(result[0].get_x(), expected, rtol=tol, atol=tol)
# ======================================================================================================================
