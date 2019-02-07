import numpy as np
from pprint import pprint


def around_borders(v, r):
    lb = v - v / 2.0
    if lb < 0.0:
        lb = 0.0
    rb = v + v / 2.0
    if rb > 1.0:
        rb = 1.0
    elif rb == 0.0:
        rb = 0.1
    return round(lb, r), round(rb, r)


def generate_s_around(num, rb, vector):
    b = around_borders
    r = str(rb/num).count("0")
    visited = set()
    for i in np.linspace(*b(vector[0], r), num, True):
        for j in np.linspace(*b(vector[1], r), num, True):
            for k in np.linspace(*b(vector[2], r), num, True):
                i, j, k = round(i, r), round(j, r), round(k, r)
                if round(i + j + k, r) == rb and (i, j, k) not in visited:
                    visited.add((i, j, k ))
                    yield (i, j, k)


if __name__ == "__main__":
    pprint(list(generate_s_around(50, 1.0, (0.0, 0.0, 1.0))))