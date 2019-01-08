from sympy import *
from scipy.optimize import minimize_scalar
from math import fabs, sqrt as math_sqrt

EPS = 0.001

if __name__ == "__main__":

    # TODO make net from 0 to N from N + 1 to 2N
    # TODO make substitution on Lambda
    # TODO find min Lambda
    # TODO make external cycle while difference between F_min >= EPS

    mu_i, K_old_i_prev, su_old_i, X1_old_prev, dt = symbols("mu_i K_old_i_prev su_old_i X1_old_prev dt")
    K_old_i = (-mu_i * K_old_i_prev + su_old_i * X1_old_prev) * dt + K_old_i_prev
    print(K_old_i)

    L_prev, nu = symbols("L_prev nu")
    L = L_prev * exp(nu * dt)
    print(L)

    theta_old_i = symbols("theta_old_i")
    L_old_i = L * theta_old_i
    print(L_old_i)

    A_old_i, alpha_old_i, betta_old_i = symbols("A_old_i alpha_old_i betta_old_i")
    X_old_i = A_old_i * L_old_i ** alpha_old_i * K_old_i ** betta_old_i
    print(X_old_i)

    st_old_i, X1_old = symbols("st_old_i X1_old")
    Ii = st_old_i * X1_old
    print(Ii)

    investments_balance = lambda su, st: fabs(sum(su) + sum(st) - 1) <= EPS

    consuming_bound = lambda X2, C: X2 >= C

    resources_balance = lambda X, a: fabs(X[0] - sum(a * X)) <= EPS

    # --------------------------------------------------------------

    K_new_i_prev, su_new_i, X1_new_prev, I_old_i, st_new_i = symbols("K_new_i_prev su_new_i X1_new_prev I_old_i st_new_i")
    K_new_i = (-mu_i * K_new_i_prev + su_new_i * Ii + st_new_i * X1_new_prev) * dt + K_new_i_prev
    print(K_new_i)

    ki = symbols("ki")
    L_new_i = K_new_i / ki
    print(L_new_i)

    # TODO calculate L
    theta_new_i = L_new_i / L

    A_new_i, alpha_new_i, betta_new_i = symbols("A_new_i alpha_new_i betta_new_i")
    X_new_i = A_new_i * L_new_i ** alpha_new_i * K_new_i ** betta_new_i

    L_old_i = (theta_old_i - theta_new_i) * L
    # TODO calculate X_old_i

    consuming_bound_new = lambda X2_new, X2_old, C: X2_new + X2_old >= C

    investments_balance_old = lambda su: fabs(sum(su) - 1) <= EPS

    investments_balance_new = lambda st: fabs(sum(st) - 1) <= EPS