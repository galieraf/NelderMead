"""
Module for representing points in n-dimensional space.
"""


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
