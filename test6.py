import json
import pickle
from itertools import chain
from time import gmtime, strftime
import numpy as np
from scipy.optimize import fmin_slsqp
from sympy import *


def scipy_f_wrap(f):
    return lambda x: np.array(f(*x))


class RearmingSimulation:
    def __init__(self):

        with open("initial_data.json") as json_file:
            initial_data = json_file.read()
            self.json_initial_data = json.loads(initial_data)

        self.C = float(self.json_initial_data["C"])
        self.ds = int(self.json_initial_data["ds"])
        self.tau = float(self.json_initial_data["tau"])
        self.N = self.tau * 2.0
        self.dt = float(self.json_initial_data["dh"])
        self.nu = float(self.json_initial_data["nu"])
        self.results = {0: {}}
        self.res0 = {}

    @staticmethod
    def xfrange(start, stop, step):
        i = 0
        while start + i * step < stop:
            yield start + i * step
            i += 1

    @staticmethod
    def generate_s(size, share):
        for i in range(0, size, 1):
            for j in range(0, size, 1):
                for k in range(0, size, 1):
                    if (i + j + k) == size * share:
                        yield (i * 1.0 / size, j * 1.0 / size, k * 1.0 / size)

    @staticmethod
    def generate_s_new(num, rb):
        # NEED TO BE VERY CAREFUL
        r = str(np.true_divide([rb], num)).count("0")
        for i in np.linspace(0.0, rb, num, True):
            for j in np.linspace(0.0, rb, num, True):
                if i + j <= rb:
                    i, j, k = round(i, r), round(j, r), round(rb - i - j, r)
                    yield (i, j, k)

    @staticmethod
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

    @staticmethod
    def generate_s_around(num, rb, vector):
        b = RearmingSimulation.around_borders
        # NEED TO BE VERY CAREFUL
        r = str(np.true_divide([rb], num)).count("0")
        visited = set()
        yield vector
        for i in np.linspace(*b(vector[0], r), num, True):
            for j in np.linspace(*b(vector[1], r), num, True):
                if i + j < rb:
                    i, j, k = round(i, r), round(j, r), round(rb - i - j, r)
                    if (i, j, k) not in visited:
                        visited.add((i, j, k))
                        yield (i, j, k)

    def init_equation_system(self):

        self.COND = {}
        self.EQ = {"L_{N0}".format(N0=0.0): self.json_initial_data["L0"],
                   "a": [float(item) for item in self.json_initial_data["a"]]}

        for i in range(0, 3):
            self.EQ["mu_{i}".format(i=i)] = float(self.json_initial_data["mu"][i])
            self.EQ["K_old_{i}_{N0}".format(i=i, N0=0.0)] = float(self.json_initial_data["K_old_0"][i])
            self.EQ["L_old_{i}_{N0}".format(i=i, N0=0.0)] = float(self.json_initial_data["L_old_0"][i])
            self.EQ["theta_old_{i}".format(i=i)] = float(self.json_initial_data["theta_old"][i])
            self.EQ["A_old_{i}".format(i=i)] = float(self.json_initial_data["A_old"][i])
            self.EQ["alpha_old_{i}".format(i=i)] = float(self.json_initial_data["alpha_old"][i])
            self.EQ["betta_old_{i}".format(i=i)] = float(self.json_initial_data["betta_old"][i])
            self.EQ["a_{i}".format(i=i)] = float(self.json_initial_data["a"][i])

        self.EQ["X_old_1_{N0}".format(N0=0.0)] = self.EQ["A_old_1"] * \
                                                 self.EQ["L_old_1_{N0}".format(N0=0.0)] ** self.EQ["alpha_old_1"] * \
                                                 self.EQ["K_old_1_{N0}".format(N0=0.0)] ** self.EQ["betta_old_1"]

        for j in self.xfrange(self.dt, self.tau + self.dt, self.dt):
            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i), negative=False)
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_old_{i}_{pN}".format(pN=j - self.dt, i=i)] +
                                                             self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_old_1_{pN}".format(pN=j - self.dt,
                                                                                           i=i)]) * self.dt + \
                                                            self.EQ["K_old_{i}_{pN}".format(pN=j - self.dt, i=i)]

            self.EQ["L_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - self.dt)] * exp(self.nu * self.dt)

            for i in range(0, 3):
                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ["L_{N}".format(N=j, i=i)] * \
                                                            self.EQ["theta_old_{i}".format(i=i)]

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * \
                                                            self.EQ["L_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_old_{i}".format(i=i)] * \
                                                            self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_old_{i}".format(i=i)]

                self.EQ["st_old_{i}_{N}".format(N=j, i=i)] = symbols("st_old_{i}_{N}".format(N=j, i=i), negative=False)

            for i in range(0, 3):
                self.EQ["I_{i}_{N}".format(N=j, i=i)] = self.EQ["st_old_{i}_{N}".format(N=j, i=i)] * \
                                                        self.EQ["X_old_1_{pN}".format(pN=j - self.dt, i=i)]

            self.COND["invest_{N}".format(N=j)] = self.EQ["su_old_0_{N}".format(N=j)] + \
                                                  self.EQ["su_old_1_{N}".format(N=j)] + \
                                                  self.EQ["su_old_2_{N}".format(N=j)] + \
                                                  self.EQ["st_old_0_{N}".format(N=j)] + \
                                                  self.EQ["st_old_1_{N}".format(N=j)] + \
                                                  self.EQ["st_old_2_{N}".format(N=j)]

            self.COND["invest_M_{N}".format(N=j)] = 1 - self.COND["invest_{N}".format(N=j)]  # >0

            self.COND["balance_{N}".format(N=j)] = (self.EQ["X_old_0_{N}".format(N=j)] -
                                                    (self.EQ["X_old_0_{N}".format(N=j)] * self.EQ["a_0"] +
                                                     self.EQ["X_old_1_{N}".format(N=j)] * self.EQ["a_1"] +
                                                     self.EQ["X_old_2_{N}".format(N=j)] * self.EQ["a_2"])) / \
                                                   self.EQ["L_{N}".format(N=j)]

            self.COND["consuming_bound_{N}".format(N=j)] = self.EQ["X_old_2_{N}".format(N=j)]

            self.COND["consuming_bound_L_{N}".format(N=j)] = self.EQ["X_old_2_{N}".format(N=j)] - self.C  # >0

        for i in range(0, 3):
            self.EQ["theta_old_{i}_{pN}".format(i=i, pN=self.tau)] = self.EQ["theta_old_{i}".format(i=i)]
            self.EQ["K_new_{i}_{pN}".format(pN=self.tau, i=i)] = 0.0
            self.EQ["L_new_{i}_{pN}".format(pN=self.tau, i=i)] = 0
            self.EQ["A_new_{i}".format(i=i)] = float(self.json_initial_data["A_new"][i])
            self.EQ["alpha_new_{i}".format(i=i)] = float(self.json_initial_data["alpha_new"][i])
            self.EQ["betta_new_{i}".format(i=i)] = float(self.json_initial_data["betta_new"][i])
            self.EQ["k_{i}".format(i=i)] = float(self.json_initial_data["k_new"][i])
            self.EQ["theta_old_{i}_tau".format(i=i)] = self.EQ["L_old_{i}_{N}".format(N=self.tau, i=i)] / self.EQ[
                "L_{N}".format(N=self.tau)]

        self.EQ["X_new_1_{pN}".format(pN=self.tau)] = 0.0

        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):
            for i in range(0, 3):
                self.EQ["st_new_{i}_{N}".format(N=j, i=i)] = symbols("st_new_{i}_{N}".format(N=j, i=i), negative=False)

                self.EQ["K_new_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_new_{i}_{pN}".format(pN=j - self.dt, i=i)] +
                                                             self.EQ["I_{i}_{pN}".format(pN=j - self.tau, i=i)] +
                                                             self.EQ["st_new_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_new_1_{pN}".format(pN=j - self.dt,
                                                                                           i=i)]) * self.dt + \
                                                            self.EQ["K_new_{i}_{pN}".format(pN=j - self.dt, i=i)]

            self.EQ["L_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - self.dt)] * exp(self.nu * self.dt)

            for i in range(0, 3):
                self.EQ["L_new_{i}_{N}".format(N=j, i=i)] = self.EQ["K_new_{i}_{N}".format(N=j, i=i)] / \
                                                            self.EQ["k_{i}".format(i=i)]

            self.COND["L_balance_{N}".format(N=j)] = self.EQ["L_{N}".format(N=j)] - \
                                                     (self.EQ["L_new_0_{N}".format(N=j)] +
                                                      self.EQ["L_new_1_{N}".format(N=j)] +
                                                      self.EQ["L_new_2_{N}".format(N=j)])  # >0

            for i in range(0, 3):
                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{N}".format(N=j, i=i)] / \
                                                                self.EQ["L_{N}".format(N=j, i=i)]

                self.EQ["X_new_{i}_{N}".format(N=j, i=i)] = self.EQ["A_new_{i}".format(i=i)] * \
                                                            self.EQ["L_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_new_{i}".format(i=i)] * \
                                                            self.EQ["K_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_new_{i}".format(i=i)]

            theta_new_sum = self.EQ["theta_new_0_{N}".format(N=j)] + \
                            self.EQ["theta_new_1_{N}".format(N=j)] + \
                            self.EQ["theta_new_2_{N}".format(N=j)]

            for i in range(0, 3):
                self.EQ["theta_old_{i}_{N}".format(i=i, N=j)] = self.EQ["theta_old_{i}_tau".format(i=i)] * \
                                                                (1 - theta_new_sum)

                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ["theta_old_{i}_{N}".format(i=i, N=j)] * \
                                                            self.EQ["L_{N}".format(N=j, i=i)]

            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i))
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_old_{i}_{pN}".format(pN=j - self.dt, i=i)] +
                                                             self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_old_1_{pN}".format(pN=j - self.dt,
                                                                                           i=i)]) * self.dt + \
                                                            self.EQ["K_old_{i}_{pN}".format(pN=j - self.dt, i=i)]

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * \
                                                            self.EQ["L_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_old_{i}".format(i=i)] * \
                                                            self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_old_{i}".format(i=i)]

            self.COND["invest_new_{N}".format(N=j)] = self.EQ["st_new_0_{N}".format(N=j)] + \
                                                      self.EQ["st_new_1_{N}".format(N=j)] + \
                                                      self.EQ["st_new_2_{N}".format(N=j)]

            self.COND["invest_new_M_{N}".format(N=j)] = 1 - self.COND["invest_new_{N}".format(N=j)]  # >0

            self.COND["invest_old_{N}".format(N=j)] = self.EQ["su_old_0_{N}".format(N=j)] + \
                                                      self.EQ["su_old_1_{N}".format(N=j)] + \
                                                      self.EQ["su_old_2_{N}".format(N=j)]

            self.COND["invest_old_M_{N}".format(N=j)] = 1 - self.COND["invest_old_{N}".format(N=j)]  # >0

            self.COND["balance_new_{N}".format(N=j)] = (self.EQ["X_new_0_{N}".format(N=j)] -
                                                        (self.EQ["X_new_0_{N}".format(N=j)] * self.EQ["a_0"] +
                                                         self.EQ["X_new_1_{N}".format(N=j)] * self.EQ["a_1"] +
                                                         self.EQ["X_new_2_{N}".format(N=j)] * self.EQ["a_2"])) / \
                                                       self.EQ["L_{N}".format(N=j)]

            self.COND["consuming_bound_{N}".format(N=j)] = self.EQ["X_new_2_{N}".format(N=j)] + \
                                                           self.EQ["X_old_2_{N}".format(N=j)]

            self.COND["consuming_bound_L_{N}".format(N=j)] = self.COND["consuming_bound_{N}".format(N=j)] - self.C  # >0

    def find_initial_vector(self):

        print(strftime("%H:%M:%S", gmtime()), "Phase 1 is started")

        for j in self.xfrange(self.dt, self.tau + self.dt, self.dt):
            s = None
            K_old_0 = lambdify(self.EQ["su_old_0_{N}".format(N=j)], self.EQ["K_old_0_{N}".format(N=j)])
            K_old_1 = lambdify(self.EQ["su_old_1_{N}".format(N=j)], self.EQ["K_old_1_{N}".format(N=j)])
            K_old_2 = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["K_old_2_{N}".format(N=j)])
            consumption = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["X_old_2_{N}".format(N=j)])
            balance = lambdify([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] for i in range(0, 3)],
                               self.COND["balance_{N}".format(N=j)])

            for S_phase_1 in self.generate_s(self.ds, 0.7):
                if K_old_0(S_phase_1[0]) > 0 and K_old_1(S_phase_1[1]) > 0 and K_old_2(S_phase_1[2]) > 0:
                    if consumption(S_phase_1[2]) >= self.C and -1.0 <= balance(*S_phase_1) <= 1.0:
                        s = S_phase_1
                        break
            if not s:
                print(strftime("%H:%M:%S", gmtime()), "nothing was found")
                break
            values = [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)]

            for k in self.xfrange(j + self.dt, self.tau + self.dt, self.dt):
                self.EQ["K_old_0_{N}".format(N=k)] = self.EQ["K_old_0_{N}".format(N=k)].subs(values)
                self.EQ["K_old_1_{N}".format(N=k)] = self.EQ["K_old_1_{N}".format(N=k)].subs(values)
                self.EQ["K_old_2_{N}".format(N=k)] = self.EQ["K_old_2_{N}".format(N=k)].subs(values)

                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].subs(
                    [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])

                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].subs(values)

            print(strftime("%H:%M:%S", gmtime()), "step {j} s: {s}".format(j=j, s=s))

            self.results[0].update({self.EQ["su_old_{i}_{N}".format(N=j, i=i)]: s[i] for i in range(0, 3)})

        print(strftime("%H:%M:%S", gmtime()), "Phase 1 is completed")
        print(strftime("%H:%M:%S", gmtime()), "Phase 2 is started")

        is_complete = True

        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):
            _st_new, _st_old, _su_old = None, None, None

            K_new_0_subs = self.EQ["K_new_0_{N}".format(N=j)].subs(self.results[0])
            K_new_0 = lambdify([self.EQ["st_old_0_{N}".format(N=j - self.tau)], self.EQ["st_new_0_{N}".format(N=j)]],
                               K_new_0_subs)

            K_new_1_subs = self.EQ["K_new_1_{N}".format(N=j)].subs(self.results[0])
            K_new_1 = lambdify([self.EQ["st_old_1_{N}".format(N=j - self.tau)], self.EQ["st_new_1_{N}".format(N=j)]],
                               K_new_1_subs)

            K_new_2_subs = self.EQ["K_new_2_{N}".format(N=j)].subs(self.results[0])
            K_new_2 = lambdify([self.EQ["st_old_2_{N}".format(N=j - self.tau)], self.EQ["st_new_2_{N}".format(N=j)]],
                               K_new_2_subs)

            L_balance_subs = self.COND["L_balance_{N}".format(N=j)].subs(self.results[0])

            L_balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                  self.EQ["st_new_1_{N}".format(N=j)],
                                  self.EQ["st_new_2_{N}".format(N=j)],
                                  self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                  self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                  self.EQ["st_old_2_{N}".format(N=j - self.tau)]], L_balance_subs)

            consumption_subs = self.COND["consuming_bound_{N}".format(N=j)].subs(self.results[0])
            consumption = lambdify((self.EQ["st_new_0_{N}".format(N=j)],
                                    self.EQ["st_new_1_{N}".format(N=j)],
                                    self.EQ["st_new_2_{N}".format(N=j)],
                                    self.EQ["su_old_2_{N}".format(N=j)],
                                    self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_2_{N}".format(N=j - self.tau)]), consumption_subs)

            balance_subs = self.COND["balance_new_{N}".format(N=j)].subs(self.results[0])
            balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                self.EQ["st_new_1_{N}".format(N=j)],
                                self.EQ["st_new_2_{N}".format(N=j)],
                                self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_2_{N}".format(N=j - self.tau)]], balance_subs)

            for st_new in self.generate_s(int(self.ds / 4), 1.0):

                for st_old in self.generate_s(self.ds, 0.3):

                    if K_new_0(st_old[0], st_new[0]) >= 0 and \
                                    K_new_1(st_old[1], st_new[1]) >= 0 and \
                                    K_new_2(st_old[2], st_new[2]) >= 0 and \
                                    L_balance(*chain(st_new, st_old)) >= 0:

                        b = balance(*chain(st_new, st_old))

                        # if abs(b) < 5:
                        #     print("b =", b, "st_old =", st_old, "st_new =", st_new)

                        if -1.0 <= round(b, 1) <= 1.0:
                            _st_new, _st_old = st_new, st_old

                            for su_old in self.generate_s(self.ds, 1.0):
                                if consumption(*chain(st_new, [su_old[2]], st_old)) >= self.C:
                                    _su_old = su_old
                                    break

                    if _su_old:
                        break

                if _su_old:
                    break

            if not _su_old:
                print("nothing was found")
                is_complete = False
                break

            print(strftime("%H:%M:%S", gmtime()),
                  "step {j} st_new: {st_new} su_old: {su_old} st_old: {st_old}".format(j=j,
                                                                                       st_new=_st_new,
                                                                                       su_old=_su_old,
                                                                                       st_old=_st_old))

            for i in range(0, 3):
                self.results[0].update({self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)]: _st_old[i]})
                self.results[0].update({self.EQ["su_old_{i}_{N}".format(i=i, N=j)]: _su_old[i]})
                self.results[0].update({self.EQ["st_new_{i}_{N}".format(i=i, N=j)]: _st_new[i]})

            l_old = [self.EQ["L_old_{i}_{N}".format(N=j, i=i)].subs(self.results[0]) for i in range(0, 3)]
            l_new = [self.EQ["L_new_{i}_{N}".format(N=j, i=i)].subs(self.results[0]) for i in range(0, 3)]

            print("L_old: {l_old} L_new: {l_new}".format(l_old=l_old, l_new=l_new))

            target_func = (self.EQ["theta_old_0"] - self.EQ["theta_new_0_{N}".format(N=j)]) ** 2 + \
                          (self.EQ["theta_old_1"] - self.EQ["theta_new_1_{N}".format(N=j)]) ** 2 + \
                          (self.EQ["theta_old_2"] - self.EQ["theta_new_2_{N}".format(N=j)]) ** 2

            print(strftime("%H:%M:%S", gmtime()), "F =", target_func.subs(self.results[0]))

        print(strftime("%H:%M:%S", gmtime()), "Phase 2 is completed")
        return is_complete

    def find_initial_vector_using_prev(self, prev_dt, prev_tau, prev_N):
        print(strftime("%H:%M:%S", gmtime()), "Phase 1 is started")

        prev_vector = self.results[len(self.results) - 1]
        s_state = {}
        generators = {}
        prev_s_state = {}

        for j in self.xfrange(prev_dt, prev_tau + prev_dt, prev_dt):
            v = tuple(prev_vector[self.EQ["su_old_{i}_{N}".format(i=i, N=j)]] for i in range(0, 3))
            b = round(sum(v), 2)
            generators[j - self.dt] = list(self.generate_s_around(self.ds, b, v))  # TODO make flexible BOUNDS !!!
            generators[j] = list(self.generate_s_around(self.ds, b, v))
            prev_s_state[j - self.dt] = v
            prev_s_state[j] = v

        is_complete = True

        self.results_next = {0: {}}

        lb = -128.0
        rb = 128.0

        for j in self.xfrange(self.dt, self.tau + self.dt, self.dt):
            s = None
            K_old_0 = lambdify(self.EQ["su_old_0_{N}".format(N=j)], self.EQ["K_old_0_{N}".format(N=j)])
            K_old_1 = lambdify(self.EQ["su_old_1_{N}".format(N=j)], self.EQ["K_old_1_{N}".format(N=j)])
            K_old_2 = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["K_old_2_{N}".format(N=j)])
            consumption = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["X_old_2_{N}".format(N=j)])
            balance = lambdify([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] for i in range(0, 3)],
                               self.COND["balance_{N}".format(N=j)])

            f_min = 1000
            for S_phase_1 in generators[j]:
                if K_old_0(S_phase_1[0]) > 0 and K_old_1(S_phase_1[1]) > 0 and K_old_2(S_phase_1[2]) > 0:
                    b = balance(*S_phase_1)
                    if lb <= b <= rb and consumption(S_phase_1[2]) >= self.C:
                        f = sum(map(lambda x: x ** 2, np.subtract(S_phase_1, prev_s_state[j]))) + abs(b)
                        if f < f_min:
                            s = S_phase_1
                            f_min = f
                            print(strftime("%H:%M:%S", gmtime()), "f_min =", f_min)
                            if f_min <= 1:
                                break
            if not s:
                print(strftime("%H:%M:%S", gmtime()), "nothing was found")
                is_complete = False
                break
            values = [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)]

            for k in self.xfrange(j + self.dt, self.tau + self.dt, self.dt):
                self.EQ["K_old_0_{N}".format(N=k)] = self.EQ["K_old_0_{N}".format(N=k)].subs(values)
                self.EQ["K_old_1_{N}".format(N=k)] = self.EQ["K_old_1_{N}".format(N=k)].subs(values)
                self.EQ["K_old_2_{N}".format(N=k)] = self.EQ["K_old_2_{N}".format(N=k)].subs(values)

                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].subs(
                    [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])

                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].subs(values)

            print(strftime("%H:%M:%S", gmtime()), "step {j} s: {s}".format(j=j, s=s))

            self.results_next[0].update({self.EQ["su_old_{i}_{N}".format(N=j, i=i)]: s[i] for i in range(0, 3)})

            s_state[j] = round(sum(s), 2)

        if not is_complete:
            print(strftime("%H:%M:%S", gmtime()), "Phase 1 isn't completed")
            return

        print(strftime("%H:%M:%S", gmtime()), "Phase 1 is completed")
        print(strftime("%H:%M:%S", gmtime()), "Phase 2 is started")

        generators = {}
        is_complete = True
        lb = -128.0
        rb = 128.0

        for j in self.xfrange(prev_tau + prev_dt, prev_N + prev_dt, prev_dt):
            v = [prev_vector[self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)]] for i in range(0, 3)]
            b = round(sum(v), 2)
            generators[j - self.dt] = list(self.generate_s_around(self.ds, b, v))
            generators[j] = list(self.generate_s_around(self.ds, b, v))

        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):
            _st_new, _st_old, _su_old = None, None, None

            K_new_0_subs = self.EQ["K_new_0_{N}".format(N=j)].subs(self.results_next[0])
            K_new_0 = lambdify([self.EQ["st_old_0_{N}".format(N=j - self.tau)], self.EQ["st_new_0_{N}".format(N=j)]],
                               K_new_0_subs)

            K_new_1_subs = self.EQ["K_new_1_{N}".format(N=j)].subs(self.results_next[0])
            K_new_1 = lambdify([self.EQ["st_old_1_{N}".format(N=j - self.tau)], self.EQ["st_new_1_{N}".format(N=j)]],
                               K_new_1_subs)

            K_new_2_subs = self.EQ["K_new_2_{N}".format(N=j)].subs(self.results_next[0])
            K_new_2 = lambdify([self.EQ["st_old_2_{N}".format(N=j - self.tau)], self.EQ["st_new_2_{N}".format(N=j)]],
                               K_new_2_subs)

            L_balance_subs = self.COND["L_balance_{N}".format(N=j)].subs(self.results_next[0])

            L_balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                  self.EQ["st_new_1_{N}".format(N=j)],
                                  self.EQ["st_new_2_{N}".format(N=j)],
                                  self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                  self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                  self.EQ["st_old_2_{N}".format(N=j - self.tau)]], L_balance_subs)

            consumption_subs = self.COND["consuming_bound_{N}".format(N=j)].subs(self.results_next[0])
            consumption = lambdify((self.EQ["st_new_0_{N}".format(N=j)],
                                    self.EQ["st_new_1_{N}".format(N=j)],
                                    self.EQ["st_new_2_{N}".format(N=j)],
                                    self.EQ["su_old_2_{N}".format(N=j)],
                                    self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_2_{N}".format(N=j - self.tau)]), consumption_subs)

            balance_subs = self.COND["balance_new_{N}".format(N=j)].subs(self.results_next[0])
            balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                self.EQ["st_new_1_{N}".format(N=j)],
                                self.EQ["st_new_2_{N}".format(N=j)],
                                self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_2_{N}".format(N=j - self.tau)]], balance_subs)

            try:
                st_new_0 = [prev_vector[self.EQ["st_new_{i}_{N}".format(i=i, N=j)]] for i in range(0, 3)]
                su_old_0 = [prev_vector[self.EQ["su_old_{i}_{N}".format(i=i, N=j)]] for i in range(0, 3)]
                st_old_0 = [prev_vector[self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)]] for i in range(0, 3)]
            except Exception as e:
                st_new_0 = [prev_vector[self.EQ["st_new_{i}_{N}".format(i=i, N=j + self.dt)]] for i in range(0, 3)]
                su_old_0 = [prev_vector[self.EQ["su_old_{i}_{N}".format(i=i, N=j + self.dt)]] for i in range(0, 3)]
                st_old_0 = [prev_vector[self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau + self.dt)]] for i in
                            range(0, 3)]

            st_new_generator = self.generate_s_around(self.ds / 2, 1.0, st_new_0)
            su_old_generator = list(self.generate_s_around(self.ds / 2, 1.0, su_old_0))

            f_min = 1000

            for st_new in st_new_generator:
                for st_old in generators[j]:
                    if K_new_0(st_old[0], st_new[0]) >= 0 and \
                                    K_new_1(st_old[1], st_new[1]) >= 0 and \
                                    K_new_2(st_old[2], st_new[2]) >= 0 and \
                                    L_balance(*chain(st_new, st_old)) >= 0:
                        b = balance(*chain(st_new, st_old))
                        if lb <= round(b, 4) <= rb:
                            for su_old in su_old_generator:
                                c = consumption(*chain(st_new, [su_old[2]], st_old)) - self.C
                                if c >= 0:
                                    f_cur = sum(map(lambda x: x ** 2, np.subtract(st_new, st_new_0))) + \
                                            sum(map(lambda x: x ** 2, np.subtract(su_old, su_old_0))) + \
                                            sum(map(lambda x: x ** 2, np.subtract(st_old, st_old_0))) + abs(b)
                                    if f_cur < f_min:
                                        _st_new, _st_old, _su_old = st_new, st_old, su_old
                                        f_min = f_cur
                                        print(strftime("%H:%M:%S", gmtime()), "f_min =", f_min)
                                        if f_min <= 1:
                                            break

            if not _su_old:
                print("nothing was found")
                is_complete = False
                break

            print(strftime("%H:%M:%S", gmtime()),
                  "step {j} st_new: {st_new} su_old: {su_old} st_old: {st_old}".format(j=j,
                                                                                       st_new=_st_new,
                                                                                       su_old=_su_old,
                                                                                       st_old=_st_old))

            for i in range(0, 3):
                self.results_next[0].update({self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)]: _st_old[i]})
                self.results_next[0].update({self.EQ["su_old_{i}_{N}".format(i=i, N=j)]: _su_old[i]})
                self.results_next[0].update({self.EQ["st_new_{i}_{N}".format(i=i, N=j)]: _st_new[i]})

            l_old = [self.EQ["L_old_{i}_{N}".format(N=j, i=i)].subs(self.results_next[0]) for i in range(0, 3)]
            l_new = [self.EQ["L_new_{i}_{N}".format(N=j, i=i)].subs(self.results_next[0]) for i in range(0, 3)]

            print("L_old: {l_old} L_new: {l_new}".format(l_old=l_old, l_new=l_new))

            target_func = (self.EQ["theta_old_0"] - self.EQ["theta_new_0_{N}".format(N=j)]) ** 2 + \
                          (self.EQ["theta_old_1"] - self.EQ["theta_new_1_{N}".format(N=j)]) ** 2 + \
                          (self.EQ["theta_old_2"] - self.EQ["theta_new_2_{N}".format(N=j)]) ** 2

            print(strftime("%H:%M:%S", gmtime()), "F =", target_func.subs(self.results_next[0]))

        print(strftime("%H:%M:%S", gmtime()), "Phase 2 is completed")
        return is_complete

    def find_min_vector(self, results):

        target_func = (self.EQ["theta_old_0"] - self.EQ["theta_new_0_{N}".format(N=self.N)]) ** 2 + \
                      (self.EQ["theta_old_1"] - self.EQ["theta_new_1_{N}".format(N=self.N)]) ** 2 + \
                      (self.EQ["theta_old_2"] - self.EQ["theta_new_2_{N}".format(N=self.N)]) ** 2

        step = 0
        f_prev = target_func.subs(results[0])
        f_current = 0
        self.labor = {}

        while True:

            for j in reversed(list(self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt))):

                search_vector = [self.EQ["st_new_{i}_{N}".format(i=i, N=j)] for i in range(0, 3)]
                if not self._part_vector(target_func, search_vector, step, results):
                    break
                step += 1

                search_vector = [self.EQ["su_old_{i}_{N}".format(i=i, N=j)] for i in range(0, 3)]
                if not self._part_vector(target_func, search_vector, step, results):
                    break
                step += 1

                search_vector = [self.EQ["su_old_{i}_{N}".format(i=i, N=j - self.tau)] for i in range(0, 3)] + \
                                [self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)] for i in range(0, 3)]
                if not self._part_vector(target_func, search_vector, step, results):
                    break
                step += 1
            else:

                f_current = target_func.subs(results[step])

                if abs(f_prev - f_current) > 0.001:
                    f_prev = f_current
                    continue

            break

        print(strftime("%H:%M:%S", gmtime()), "Optimization complete. Final results are:")

        for k, v in results[step].items():
            print(k, "=", v)

        print("F =", target_func.subs(results[step]))

        self.labor[0] = {}
        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):
            self.labor[0].update({"L_{N}".format(N=j): str(self.EQ["L_{N}".format(N=j)])})

            self.labor[0].update({"L_new_{i}_{N}".format(N=j, i=i):
                                      str(self.EQ["L_new_{i}_{N}".format(N=j, i=i)].subs(results[step])) for i in
                                  range(0, 3)})
            self.labor[0].update({"L_old_{i}_{N}".format(N=j, i=i):
                                      str(self.EQ["L_old_{i}_{N}".format(N=j, i=i)].subs(results[step])) for i in
                                  range(0, 3)})

            self.labor[0].update({"theta_new_{i}_{N}".format(N=j, i=i):
                                      str(self.EQ["theta_new_{i}_{N}".format(N=j, i=i)].subs(results[step])) for i in
                                  range(0, 3)})
            self.labor[0].update({"theta_old_{i}_{N}".format(N=j, i=i):
                                      str(self.EQ["theta_old_{i}_{N}".format(N=j, i=i)].subs(results[step])) for i in
                                  range(0, 3)})

    def _part_vector(self, target_func, search_vector, step, results):

        subs_vector = {k: v for k, v in results[step].items() if k not in search_vector}

        objective = scipy_f_wrap(lambdify(search_vector, target_func.subs(subs_vector)))

        init_vector = [results[step][s] for s in search_vector]

        ieqcons_list = []
        eqcons_list = []
        COND = {}
        bounds_x = []

        for i in range(0, len(search_vector)):
            ieqcons_list.append(lambda x, i=i: x[i])
            COND[" >= 0 X%d" % i] = lambda x, i=i: x[i]
            bounds_x.append((0, 1))

        for j in self.xfrange(self.dt, self.tau + self.dt, self.dt):

            cond_list = (
                ("== 0 invest_M_{N}".format(N=j), self.COND["invest_M_{N}".format(N=j)].subs(subs_vector)),
                ("== 0 balance_{N}".format(N=j), self.COND["balance_{N}".format(N=j)].subs(subs_vector)),
            )

            for name, cond in cond_list:
                if len(cond.free_symbols) > 0:
                    f = scipy_f_wrap(lambdify(search_vector, cond))
                    eqcons_list.append(f)
                    COND[name] = f

            cond = self.COND["consuming_bound_L_{N}".format(N=j)].subs(subs_vector)

            if len(cond.free_symbols) > 0:
                f = scipy_f_wrap(lambdify(search_vector, cond))
                ieqcons_list.append(f)
                COND[" >= 0 consuming_bound_L_{N}".format(N=j)] = f

        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):

            cond_list = (
                ("== 0 invest_old_M_{N}".format(N=j), self.COND["invest_old_M_{N}".format(N=j)].subs(subs_vector)),
                ("== 0 invest_new_M_{N}".format(N=j), self.COND["invest_new_M_{N}".format(N=j)].subs(subs_vector)),
                ("== 0 balance_new_{N}".format(N=j), self.COND["balance_new_{N}".format(N=j)].subs(subs_vector))
            )

            for name, cond in cond_list:
                if len(cond.free_symbols) > 0:
                    f = scipy_f_wrap(lambdify(search_vector, cond))
                    eqcons_list.append(f)
                    COND[name] = f

            cond = self.COND["consuming_bound_L_{N}".format(N=j)].subs(subs_vector)

            if len(cond.free_symbols) > 0:
                f = scipy_f_wrap(lambdify(search_vector, cond))
                ieqcons_list.append(f)
                COND[" >= 0 consuming_bound_L_{N}".format(N=j)] = f

            cond = self.COND["L_balance_{N}".format(N=j)].subs(subs_vector)

            if len(cond.free_symbols) > 0:
                f = scipy_f_wrap(lambdify(search_vector, cond))
                ieqcons_list.append(f)
                COND[" >= 0 L_balance_{N}".format(N=j)] = f

        min_vector = fmin_slsqp(func=objective,
                                x0=np.array(init_vector),
                                eqcons=eqcons_list,
                                ieqcons=ieqcons_list,
                                bounds=bounds_x,
                                iter=100,
                                acc=0.1)

        if np.isnan(min_vector[1]):
            return False

        results[step + 1] = {k: v for k, v in results[step].items()}

        for i, s in enumerate(search_vector):
            results[step + 1].update({s: min_vector[0][i]})
            print(s, "=", min_vector[0][i])

        for name, cond in COND.items():
            print(cond(min_vector[0]), name)

        return True

    def save_pickle(self, results, fname):
        with open('%s.pickle' % fname, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(strftime("%H:%M:%S", gmtime()), "Saved result vector to file %s" % fname)

    def save_json(self, results, fname):
        with open('%s.json' % fname, 'w') as handle:
            json.dump({str(k): {str(nk): nv for nk, nv in v.items()}
                       for k, v in results.items()}, handle, ensure_ascii=False)
        print(strftime("%H:%M:%S", gmtime()), "Saved result vector to file %s" % fname)

    def load(self, results, fname):
        with open('%s.pickle' % fname, 'rb') as handle:
            results = pickle.load(handle)
        print(strftime("%H:%M:%S", gmtime()), "Loaded result vector from file")


if __name__ == "__main__":
    rs = RearmingSimulation()
    rs.dt = 1.0
    rs.init_equation_system()
    if rs.find_initial_vector():
        rs.find_min_vector(rs.results)
        # rs.save_pickle(rs.results, "tau2N4dt1")
        rs.save_json(rs.results, "tau2N4dt1")
        rs.save_json(rs.labor, "labor_tau2N4dt1")

        rs.dt = 0.5
        rs.init_equation_system()
        # TODO use division for next init vector and pass it to minimization function
        if rs.find_initial_vector_using_prev(1.0, 2.0, 4.0):
            rs.find_min_vector(rs.results_next)
            # rs.save_pickle(rs.results_next, "tau2N4dt05")
            rs.save_json(rs.results_next, "tau2N4dt05")
            rs.save_json(rs.labor, "labor_tau2N4dt05")

            rs.dt = 0.25
            rs.init_equation_system()
            # TODO use division for next init vector and pass it to minimization function
            rs.results = rs.results_next
            if rs.find_initial_vector_using_prev(0.5, 2.0, 4.0):
                rs.find_min_vector(rs.results_next)
                # rs.save_pickle(rs.results_next, "tau2N4dt05")
                rs.save_json(rs.results_next, "tau2N4dt025")
                rs.save_json(rs.labor, "labor_tau2N4dt025")
