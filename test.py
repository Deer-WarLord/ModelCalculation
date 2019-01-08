from sympy import *
from scipy.optimize import minimize_scalar
from math import fabs
from sympy.plotting import plot3d, plot3d_parametric_line

if __name__ == "__main__":
    eps = 0.01
    x1, x2 = symbols("x1 x2")
    func = 4*x1**2 - 10*x2**2 + 10*x1*x2 - 26*x1
    X_prev = (0, 0)
    f_prev = func.subs([(x1, X_prev[0]), (x2, X_prev[1])])
    series = [(X_prev[0], X_prev[1], f_prev)]


    while True:
        without_x2 = func.subs(x2, X_prev[1])
        extr_sign = -1 if diff(without_x2, x1, 2) < 0 else 1
        x1_next = minimize_scalar(lambdify(x1, extr_sign * without_x2)).x

        without_x1 = func.subs(x1, x1_next)
        extr_sign = -1 if diff(without_x1, x2, 2) < 0 else 1
        x2_next = minimize_scalar(lambdify(x2, extr_sign * without_x1)).x

        X_next = (x1_next, x2_next)

        f_next = func.subs([(x1, X_next[0]), (x2, X_next[1])])

        if fabs(f_prev - f_next) < eps:
            break

        f_prev = f_next
        X_prev = X_next
        series.append((X_prev[0], X_prev[1], f_prev))

    print("X", X_next)
    print("Fmin", f_next)
    series.append((X_next[0], X_next[1], f_next))

    p = plot3d(func, (x1, -100, 100), (x2, -100, 100), show=False)
    p.show()




