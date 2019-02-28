import json

import numpy as np
from pprint import pprint


def generate_s_new(num, rb):
    # NEED TO BE VERY CAREFUL
    r = str(np.true_divide([rb], num)).count("0")
    for i in np.linspace(0.0, rb, num, True):
        for j in np.linspace(0.0, rb, num, True):
            if i + j <= rb:
                i, j, k = round(i, r), round(j, r), round(rb - i - j, r)
                yield (i, j, k)


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
    r = str(np.true_divide([rb], num)).count("0")
    visited = set()
    for i in np.linspace(*b(vector[0], r), num, True):
        for j in np.linspace(*b(vector[1], r), num, True):
            if i + j <= rb:
                i, j, k = round(i, r), round(j, r), round(rb - i - j, r)
                if (i, j, k) not in visited:
                    visited.add((i, j, k))
                    yield (i, j, k)


if __name__ == "__main__":
    r = list(generate_s_around(100, 0.55, (0.536460867, 9.10188E-17, 0.015543153)))
    pprint(len(r))
    pprint(r)
    # with open("tau2N4dt1.json") as json_file:
    #     data = json_file.read()
    #     r1 = json.loads(data)
    #
    # with open("tau2N4dt05.json") as json_file:
    #     data = json_file.read()
    #     r2 = json.loads(data)

