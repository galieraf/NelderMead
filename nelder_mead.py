"""
Nelder-Mead Algorithm implementation
https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
"""

import math
import threading
Debugging = False
# timeout for infinite loops detection
time_out = 5
# coefficients
gamma = 2.0  # extension coef
beta = 0.5  # compression coef
alpha = 1.0  # reflection coef


# ----------------------------------------------------------------------------------------------------------------------
def parabola_0(x):
    """
    x^2 function

    :param x: vector with point coordinates
    :return: function value at point
    """

    r = x[0] * x[0]
    return r


# ----------------------------------------------------------------------------------------------------------------------
def parabola_1(x):
    """
    (x-1)^2 function

    :param x: vector with point coordinates
    :return: function value at point
    """

    r = (x[0] - 1) ** 2
    return r


# ----------------------------------------------------------------------------------------------------------------------
def parabola_2(x):
    """
    (x+3)^2 + 2 function

    :param x: vector with point coordinates
    :return: function value at point
    """

    r = (x[0] + 3) ** 2 + 2
    return r


# ----------------------------------------------------------------------------------------------------------------------
def ff_dim_2(x):
    """
    x^3+8y^3-6xy+1 function

    :param x: vector with point coordinates
    :return: function value at point
    """
    r = x[0] ** 3 + 8 * x[1] ** 3 - 6 * x[0] * x[1] + 1
    return r


# ----------------------------------------------------------------------------------------------------------------------
def parabaloid(x):
    """
    x^2 + y^2 function

    :param x: vector with point coordinates
    :return: function value at point
    """

    r = x[0] * x[0] + x[1] * x[1]
    return r


# ----------------------------------------------------------------------------------------------------------------------
def rosenbrock(x):
    """
    Rosenbrock function - https://en.wikipedia.org/wiki/Rosenbrock_function

    :param x: vector with point coordinates
    :return: function value at point
    """

    z1 = 1 - x[0]
    z2 = x[1] - x[0] * x[0]
    r = z1 * z1 + 100 * z2 * z2
    return r


# ----------------------------------------------------------------------------------------------------------------------
def easy_3_dim(x):
    """
    x^2 + y^2 + z^2 function

    :param x: vector with point coordinates
    :return: function value at point
    """
    z1 = x[0] ** 2
    z2 = x[1] ** 2
    z3 = x[2] ** 2
    r = z1 + z2 + z3
    return r


# ----------------------------------------------------------------------------------------------------------------------
def harder_3_dim(x):
    """
    (x^2+y^2-4)^2+(x^2+z^2-4)^2+(y^2+z^2-4)^2 function

    :param x: vector with point coordinates
    :return: function value at point
    """
    z1 = x[0] ** 2
    z2 = x[1] ** 2
    z3 = x[2] ** 2
    r = (z1 + z2 - 4) ** 2 + (z2 + z3 - 4) ** 2 + (z1 + z3 - 4) ** 2
    return r


# ======================================================================================================================
class Point:
    """
    Class represents a point in n-dimensional space
    """
    def __init__(self, vector, n):
        """
        Constructor

        :param vector: vector with point coordinates
        :param n: dimension
        """

        self.n = n
        self.vector = []
        for i in range(n):
            self.vector.append(vector[i])

    def __str__(self):
        """
        Converts to string

        :return: string with point coordinates
        """
        s = ""
        for i in range(self.n):
            s = s + str(self.vector[i]) + " "
        return s

    def __add__(self, y):
        """
        Sums two points

        :param y: other point
        :return: result point
        """

        r = []
        for i in range(self.n):
            r.append(self.vector[i] + y.vector[i])
        return Point(r, self.n)

    def __sub__(self, y):
        """
        Subtract two points

        :param y: other point
        :return: result point
        """

        r = []
        for i in range(self.n):
            r.append(self.vector[i] - y.vector[i])
        return Point(r, self.n)

    def __mul__(self, m):
        """
        Multiplies point by a scalar

        :param m: scalar
        :return: result point
        """

        r = []
        for i in range(self.n):
            r.append(self.vector[i] * m)
        return Point(r, self.n)

    def __truediv__(self, m):
        """
        Divides point by a scalar

        :param m: scalar
        :return: result point
        """

        r = []
        for i in range(self.n):
            r.append(self.vector[i] / m)
        return Point(r, self.n)

    def get_x(self):
        """
        Gets a vector with point coordinates
        :return: vector
        """

        return self.vector


# ======================================================================================================================
def zero_point(n):
    """
    Creates a n size zero vector - zero point in n-dimensional space
    :param n: dimension
    :return: zero point
    """

    x = []
    for _ in range(n):
        x.append(0.0)
    return Point(x, n)


# ----------------------------------------------------------------------------------------------------------------------
def start_simplex(edge, StartPoint, N):
    """
    Creates a simplex in n-dimensional space

    :param edge: initial simplex edge length
    :param StartPoint: initial point
    :param N: dimension - 1
    :return: vector with simplex vertices coordinates
    """

    tnsq = edge / N / math.sqrt(2.0)  # these are formulas from Himmelblau
    n1 = math.sqrt(N + 1)
    d1 = tnsq * (n1 + N - 1)
    d2 = tnsq * (n1 - 1)
    r = [Point(StartPoint, N)]  # each r[i] - Point of the polygon
    for i in range(1, N + 1):
        s = []
        for j in range(N):
            s.append(d2 + StartPoint[j])
        s[i - 1] = d1 + StartPoint[i - 1]
        r.append(Point(s + StartPoint, N))
    return r


# ======================================================================================================================
class Simplex:
    """
    Class represents simplex in n-dimensional space
    """
    def __init__(self, StartPoint, N, edge, F):
        """
        Constructor

        :param StartPoint: vector with initial point coordinates
        :param N: dimension - 1
        :param edge: initial simplex edge length
        :param F: function to be minimized
        """

        self.N = N
        self.F = F
        self.pnt = start_simplex(edge, StartPoint, N)
        self.ff = []  # function values at the vertices of the polyhedron
        for i in range(N + 1):
            self.ff.append(self.F(self.pnt[i].vector))

    def get_vertex(self, index):
        """
        Gets the vertex of the simplex at the given index.

        :param index: the index of the vertex of the simplex to get
        :return: the simplex vertex object at the specified index
        """

        return self.pnt[index]

    def get_dimension(self):
        """
        Gets dimension
        :return: dimension
        """

        return self.N
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def print_smp(s, smp):
    """
    Debugging printing

    :param s: helping string
    :param smp: a simplex to print
    :return: None
    """

    if not Debugging:
        return
    print(s)
    for i in range(smp.N + 1):
        print(smp.pnt[i].vector)
    print('')


# ----------------------------------------------------------------------------------------------------------------------
def get_extrema_indices(smp, N):
    """
    Finds indexes of vertices with minimum imin, maximum imax and
    second largest imax2 function value

    :param smp: polygon
    :param N: dimension
    :return: list of imin - [0], imax - [1], imax2 -[2]
    """
    imin, imax, imax2 = 0, 0, 0
    for i in range(N + 1):
        if smp.ff[i] < smp.ff[imin]:
            imin = i
        if smp.ff[i] >= smp.ff[imax]:
            imax2 = imax
            imax = i
    return [imin, imax, imax2]


# ----------------------------------------------------------------------------------------------------------------------
def update_simplex(smp, imax, new_point, new_value):
    """
    Update the simplex by replacing the point with the highest function value
    with a new point and its corresponding function value.

    :param smp: The simplex object
    :param imax: The index of the point with the highest function value
    :param new_point: The new point to replace the point with the highest function value
    :param new_value: The function value at the new point
    :return: None
    """
    smp.pnt[imax] = new_point
    smp.ff[imax] = new_value


# ----------------------------------------------------------------------------------------------------------------------
def expansion(F, smp, min_max_list, ph, pc):
    """
    Expansion step of the Nelder-Mead algorithm.

    :param F: function to be minimized
    :param smp: simplex
    :param min_max_list: list of indices of vertices with minimum, maximum, and second largest function values
    :param ph: vertex with the highest function value
    :param pc: centroid of all vertices except the vertex with the maximum function value
    :return: None
    """
    pr = pc*(1+alpha) - ph*alpha
    fr = F(pr.vector)

    pe = pc*(1-gamma)+pr*gamma
    fe = F(pe.vector)
    if fe < fr:
        update_simplex(smp, min_max_list[1], pe, fe)
    else:
        update_simplex(smp, min_max_list[1], pr, fr)

    print_smp('expansion ', smp)


# ----------------------------------------------------------------------------------------------------------------------
def middle_case(F, smp, min_max_list, ph, pc):
    """
    Handling the middle case in the Nelder-Mead algorithm when the reflection point has a function value between
    the lowest and the second-largest function values of the vertices.

    :param F: function to be minimized
    :param smp: simplex
    :param min_max_list: list of indices of vertices with minimum, maximum, and second-largest function values
    :param ph: vertex with the highest function value
    :param pc: centroid of all vertices except the vertex with the maximum function value
    :return: None
    """
    pr = pc*(1+alpha) - ph*alpha
    fr = F(pr.vector)

    update_simplex(smp, min_max_list[1], pr, fr)
    print_smp('fr in the middle', smp)


# ----------------------------------------------------------------------------------------------------------------------
def compression(F, N, smp, min_max_list, pc):
    """
    Compression step of the Nelder-Mead algorithm.

    :param F: function to be minimized
    :param N: number of function arguments
    :param smp: simplex
    :param min_max_list: list of indices of vertices with minimum, maximum, and second-largest function values
    :param pc: centroid of all vertices except the vertex with the maximum function value
    :return: None
    """
    ph = smp.pnt[min_max_list[1]]  # vertex with the highest function value
    ps = smp.pnt[min_max_list[1]]*beta+pc*(1-beta)
    fs = F(ps.vector)

    if fs < F(ph.vector):
        update_simplex(smp, min_max_list[1], ps, fs)
    else:
        pl = smp.pnt[min_max_list[0]]
        for i in range(N+1):
            if i != min_max_list[0]:
                smp.pnt[i] = pl+(smp.pnt[i]-pl)/2

    print_smp('compression ', smp)


# ----------------------------------------------------------------------------------------------------------------------
def one_step(F, N, smp):
    """
    One step of the Nelder-Mead algorithm - compression, expansion, or handling the middle case.

    :param F: function to be minimized
    :param N: number of function arguments
    :param smp: simplex
    :return: None
    """

    # Finds indexes of vertices with minimum imin, maximum imax and
    # second largest imax2 function value
    min_max_list = get_extrema_indices(smp, N)

    ph = smp.pnt[min_max_list[1]]
    # fh = F(ph.vector)

    pg = smp.pnt[min_max_list[2]]
    fg = F(pg.vector)

    pl = smp.pnt[min_max_list[0]]
    fl = F(pl.vector)

    # centroid of all vertices except imax
    pc = sum((smp.pnt[i] for i in range(N + 1) if i != min_max_list[1]), zero_point(N)) / N

    pr = pc*(1+alpha) - ph*alpha
    fr = F(pr.vector)

    if fr < fl:  # expansion, stage 4 on wiki
        expansion(F, smp, min_max_list, ph, pc)
    elif fl < fr < fg:  # fr in the middle, stage 5 on wiki
        middle_case(F, smp, min_max_list, ph, pc)
    else:  # compression, stage 6 on wiki
        compression(F, N, smp, min_max_list, pc)


# ----------------------------------------------------------------------------------------------------------------------
def sigma_check(smp):
    """
    https://en.wikipedia.org/wiki/Standard_deviation
    Accuracy tolerance check - calculates the standard deviation from the center of gravity

    :param smp: simplex (polygon)
    :return: point - center of gravity, standard deviation
    """
    # the center of gravity
    p = zero_point(smp.N)
    for i in range(smp.N + 1):
        p = p + smp.pnt[i]
    p = p / (smp.N + 1)

    sigma = 0
    for i in range(smp.N + 1):
        t = smp.pnt[i] - p

        s = 0
        for j in range(smp.N):
            s = s + t.vector[j] * t.vector[j]

        sigma = sigma + s
    sigma = math.sqrt(sigma / (smp.N + 1))
    return [p, sigma]


# ----------------------------------------------------------------------------------------------------------------------
def nelder_mead(fun, n, StartPoint, edge, epsilon):
    """
    The main function
    
    :param fun: a function to be minimized
    :param n: quantity of fun variables
    :param StartPoint: vector with initial point coordinates
    :param edge: initial simplex edge size
    :param epsilon: tolerance
    :return: vector with extremum point coordinates
    """

    smp = Simplex(StartPoint, n, edge, fun)
    print_smp('starting ', smp)

    result = {'value': None, 'finished': False}

    def compute_result():
        """
        A helper function for the Nelder-Mead method that computes the minimum of the provided function.

        This function computes the result of the Nelder-Mead method by iterating through the method's steps
        until the specified tolerance is reached. If an OverflowError or TypeError occurs during computation,
        the function will set the result value to -1 and mark the computation as finished.

        :return: None. The result is stored in the 'result' dictionary with keys 'value' and 'finished'.
                 The 'value' key holds the extremum point coordinates or -1 if an error occurs.
                 The 'finished' key is a boolean flag indicating whether the computation has finished or not.
        """
        try:
            while True:
                s = sigma_check(smp)
                sgm = s[1]
                if sgm < epsilon:
                    break

                one_step(fun, n, smp)

            result['value'] = smp.pnt
            result['finished'] = True
        except (OverflowError, TypeError):
            # Numerical result error
            result['value'] = -1
            result['finished'] = True

    computation_thread = threading.Thread(target=compute_result, daemon=True)
    computation_thread.start()

    computation_thread.join(timeout=time_out)

    if not result['finished']:
        # Computation took too long
        return -1

    return result['value']
# ----------------------------------------------------------------------------------------------------------------------
