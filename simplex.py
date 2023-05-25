"""
Module for representing simplex in n-dimensional space
"""
import math
import point as pt


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
    r = [pt.Point(StartPoint, N)]  # each r[i] - Point of the polygon
    for i in range(1, N + 1):
        s = []
        for j in range(N):
            s.append(d2 + StartPoint[j])
        s[i - 1] = d1 + StartPoint[i - 1]
        r.append(pt.Point(s + StartPoint, N))
    return r
# ----------------------------------------------------------------------------------------------------------------------
