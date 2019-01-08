from sympy import *
from scipy.optimize import minimize_scalar
from math import fabs, sqrt as math_sqrt
from sympy.plotting import plot3d

if __name__ == "__main__":
    eps = 0.1
    x1, x2, l = symbols("x1 x2 l")
    func = 10*x1**2 + x2**2
    X = (x1, x2)
    Xk = (10, 10)

    iteration = 1
    grad_func = [diff(func, x_i) for x_i in X]
    grad_func_eval = [grad_func[i].subs(zip(X, Xk)) for i in range(len(X))]
    print(grad_func_eval)
    res = math_sqrt(sum([val**2 for val in grad_func_eval]))
    print(res)
    print("Iteration: ", iteration)

    while res > eps:
        iteration += 1
        X_next = [Xk[i] - l * grad_func_eval[i] for i in range(len(X))]
        l_min = minimize_scalar(lambdify(l, func.subs(zip(X, X_next)))).x
        print("Step: ", l_min)

        X_next = [Xk[i] - l_min * grad_func_eval[i] for i in range(len(X))]
        print("Xk+1: ", X_next)

        grad_func_eval = [grad_func[i].subs(zip(X, X_next)) for i in range(len(X))]

        res = math_sqrt(sum([val**2 for val in grad_func_eval]))
        print(res)
        Xk = X_next
        print("Iteration: ", iteration)

    # p = plot3d(func, (x1, -100, 100), (x2, -100, 100), show=False)
    # p.show()