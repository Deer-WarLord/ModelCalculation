import json
from itertools import chain
from time import gmtime, strftime
import numpy as np
from scipy.optimize import fmin_slsqp

from sympy import *

EPS = 0.001


def scipy_f_wrap(f):
    """
        Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda x: np.array(f(*x))


class RearmingSimulation:
    def __init__(self):

        with open("initial_data.json") as json_file:
            initial_data = json_file.read()
            self.json_initial_data = json.loads(initial_data)

        self.C = float(self.json_initial_data["C"])
        self.ds = int(1.0 / float(self.json_initial_data["ds"]))
        self.tau = self.json_initial_data["tau"]
        self.N = self.tau * 2
        self.results = {0: {}}
        self.res0 = {}
        self.COND = {}
        self.EQ = {}

    @staticmethod
    def generate_s(size, share):
        for i in range(0, size, 1):
            for j in range(0, size, 1):
                for k in range(0, size, 1):
                    if (i + j + k) == size * share:
                        yield (i * 1.0 / size, j * 1.0 / size, k * 1.0 / size)

    @staticmethod
    def complex2float(val):
        return val.real if isinstance(val, complex) else val

    def init_equation_system(self):

        dt = float(self.json_initial_data["dh"])
        nu = float(self.json_initial_data["nu"])
        self.EQ["L_0"] = self.json_initial_data["L0"]
        self.EQ["a"] = [float(item) for item in self.json_initial_data["a"]]

        for i in range(0, 3):
            self.EQ["mu_{i}".format(i=i)] = float(self.json_initial_data["mu"][i])
            self.EQ["K_old_{i}_0".format(i=i)] = float(self.json_initial_data["K_old_0"][i])
            self.EQ["L_old_{i}_0".format(i=i)] = float(self.json_initial_data["L_old_0"][i])
            self.EQ["theta_old_{i}".format(i=i)] = float(self.json_initial_data["theta_old"][i])
            self.EQ["A_old_{i}".format(i=i)] = float(self.json_initial_data["A_old"][i])
            self.EQ["alpha_old_{i}".format(i=i)] = float(self.json_initial_data["alpha_old"][i])
            self.EQ["betta_old_{i}".format(i=i)] = float(self.json_initial_data["betta_old"][i])
            self.EQ["a_{i}".format(i=i)] = float(self.json_initial_data["a"][i])

        self.EQ["X_old_1_0"] = self.EQ["A_old_1"] * self.EQ["L_old_1_0"] ** self.EQ["alpha_old_1"] * \
                               self.EQ["K_old_1_0"] ** self.EQ["betta_old_1"]

        for j in range(1, self.tau + 1):
            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i), negative=False)
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                             self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_old_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                            self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)]

            self.EQ["L_{N}".format(N=j, i=i)] = self.EQ["L_{pN}".format(pN=j - 1, i=i)] * exp(nu * dt)

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
                                                        self.EQ["X_old_1_{pN}".format(pN=j - 1, i=i)]

            self.COND["invest_{N}".format(N=j)] = sum([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] +
                                                       self.EQ["st_old_{i}_{N}".format(N=j, i=i)] for i in
                                                       range(0, 3)])

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
            self.EQ["I_{i}_{tau}".format(tau=self.tau + 1, i=i)] = 0.0
            self.EQ["st_old_{i}_{tau}".format(tau=self.tau + 1, i=i)] = symbols("st_old_{i}_{tau}".format(
                tau=self.tau + 1, i=i))
            self.EQ["A_new_{i}".format(i=i)] = float(self.json_initial_data["A_new"][i])
            self.EQ["alpha_new_{i}".format(i=i)] = float(self.json_initial_data["alpha_new"][i])
            self.EQ["betta_new_{i}".format(i=i)] = float(self.json_initial_data["betta_new"][i])
            self.EQ["k_{i}".format(i=i)] = float(self.json_initial_data["k_new"][i])

        self.EQ["X_new_1_{pN}".format(pN=self.tau)] = 0.0

        for j in range(self.tau + 1, self.N + 1):
            for i in range(0, 3):
                self.EQ["st_new_{i}_{N}".format(N=j, i=i)] = symbols("st_new_{i}_{N}".format(N=j, i=i), negative=False)

                self.EQ["K_new_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_new_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                             self.EQ["I_{i}_{pN}".format(pN=j - self.tau, i=i)] +
                                                             self.EQ["st_new_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_new_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                            self.EQ["K_new_{i}_{pN}".format(pN=j - 1, i=i)]

            self.EQ["L_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - 1)] * exp(nu * dt)

            for i in range(0, 3):
                self.EQ["L_new_max_{i}_{N}".format(N=j, i=i)] = self.EQ["K_new_{i}_{N}".format(N=j, i=i)] / \
                                                                self.EQ["k_{i}".format(i=i)]

                self.EQ["L_new_real_{i}_{N}".format(N=j, i=i)] = self.EQ["theta_old_{i}_{pN}".format(i=i, pN=j - 1)] * \
                                                                 self.EQ["L_{N}".format(N=j)]

                self.EQ["L_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{pN}".format(pN=j - 1, i=i)] + \
                                                            Min(self.EQ["L_new_max_{i}_{N}".format(N=j, i=i)] -
                                                                self.EQ["L_new_{i}_{pN}".format(pN=j - 1, i=i)],
                                                                self.EQ["L_new_real_{i}_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{N}".format(N=j, i=i)] / \
                                                                self.EQ["L_{N}".format(N=j, i=i)]

                self.EQ["X_new_{i}_{N}".format(N=j, i=i)] = self.EQ["A_new_{i}".format(i=i)] * \
                                                            self.EQ["L_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_new_{i}".format(i=i)] * \
                                                            self.EQ["K_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_new_{i}".format(i=i)]

            for i in range(0, 3):
                self.EQ["theta_old_{i}_{N}".format(i=i, N=j)] = self.EQ["theta_old_{i}_{pN}".format(i=i, pN=j - 1)] - \
                                                                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)]

                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ["theta_old_{i}_{N}".format(i=i, N=j)] * \
                                                            self.EQ["L_{N}".format(N=j, i=i)]

            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i))
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                             self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_old_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                            self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)]

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * \
                                                            self.EQ["L_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_old_{i}".format(i=i)] * \
                                                            self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_old_{i}".format(i=i)]

            self.COND["invest_new_{N}".format(N=j)] = sum([self.EQ["st_new_{i}_{N}".format(N=j, i=i)] for i in
                                                           range(0, 3)])

            self.COND["invest_new_M_{N}".format(N=j)] = 1 - self.COND["invest_new_{N}".format(N=j)]  # >0

            self.COND["invest_old_{N}".format(N=j)] = sum([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] for i in
                                                           range(0, 3)])

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

        for j in range(1, self.tau + 1):
            s = None
            consumption = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["X_old_2_{N}".format(N=j)])
            balance = lambdify([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] for i in range(0, 3)],
                               self.COND["balance_{N}".format(N=j)])
            for S_phase_1 in self.generate_s(self.ds, 0.95):
                if consumption(S_phase_1[2]) >= self.C and -1.0 <= balance(*S_phase_1) <= 1.0:
                    s = S_phase_1
                    break
            if not s:
                print(strftime("%H:%M:%S", gmtime()), "nothing was found")
                break
            values = [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)]
            for k in range(j + 1, self.tau + 1):
                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].subs(
                    [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])
                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].subs(values)
            print(strftime("%H:%M:%S", gmtime()), "step {j} s: {s}".format(j=j, s=s))
            self.results[0].update({self.EQ["su_old_{i}_{N}".format(N=j, i=i)]: s[i] for i in range(0, 3)})

        print(strftime("%H:%M:%S", gmtime()), "Phase 1 is completed")
        print(strftime("%H:%M:%S", gmtime()), "Phase 2 is started")

        for j in range(self.tau + 1, self.N + 1):
            _st_new, _st_old, _su_old = None, None, None
            consumption_subs = self.COND["consuming_bound_{N}".format(N=j)].subs(self.results[0])
            consumption = lambdify((self.EQ["st_new_2_{N}".format(N=j)],
                                    self.EQ["su_old_2_{N}".format(N=j)],
                                    self.EQ["st_old_2_{tau}".format(tau=j - self.tau)]), consumption_subs)

            balance_subs = self.COND["balance_new_{N}".format(N=j)].subs(self.results[0])
            balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                self.EQ["st_new_1_{N}".format(N=j)],
                                self.EQ["st_new_2_{N}".format(N=j)],
                                self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_2_{N}".format(N=j - self.tau)]], balance_subs)

            for st_new in self.generate_s(int(self.ds / 4), 1.0):

                for st_old in self.generate_s(self.ds * 5, 0.05):

                    b = balance(*chain(st_new, st_old))

                    # if abs(b) < 10:
                    #     print("b =", b, "st_old =", st_old, "st_new =", st_new)

                    if -1.0 <= round(b, 1) <= 1.0:
                        _st_new, _st_old = st_new, st_old

                        for su_old in self.generate_s(self.ds, 1.0):
                            if self.complex2float(consumption(st_new[2], su_old[2], st_old[2])) >= self.C:
                                _su_old = su_old
                                break

                    if _su_old:
                        break

                if _su_old:
                    break

            if not _su_old:
                print("nothing was found")
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

            for k in range(j + 1, self.N + 1):
                self.COND["consuming_bound_{N}".format(N=k)] = \
                    self.COND["consuming_bound_{N}".format(N=k)].subs(self.results[0])
                self.COND["balance_new_{N}".format(N=k)] = \
                    self.COND["balance_new_{N}".format(N=k)].subs(self.results[0])

            target_func = sum([(self.EQ["theta_old_{i}".format(i=i)] -
                                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)]) for i in range(0, 3)])

            print(strftime("%H:%M:%S", gmtime()), "F =", target_func.subs(self.results[0]))

        print(strftime("%H:%M:%S", gmtime()), "Phase 2 is completed")

    def find_min_vector(self):

        target_func = sum([(self.EQ["theta_old_{i}".format(i=i)] -
                            self.EQ["theta_new_{i}_{N}".format(N=self.N, i=i)]) for i in range(0, 3)])

        step = 0
        f_prev = target_func.subs(self.results[0])
        f_current = 0

        while True:

            for j in reversed(range(self.tau + 1, self.N + 1)):

                search_vector = [self.EQ["st_new_{i}_{N}".format(i=i, N=j)] for i in range(0, 3)]
                if not self._part_vector(target_func, search_vector, step):
                    break
                step += 1

                search_vector = [self.EQ["su_old_{i}_{N}".format(i=i, N=j)] for i in range(0, 3)]
                if not self._part_vector(target_func, search_vector, step):
                    break
                step += 1

                search_vector = [self.EQ["su_old_{i}_{N}".format(i=i, N=j - self.tau)] for i in range(0, 3)] + \
                                [self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)] for i in range(0, 3)]
                if not self._part_vector(target_func, search_vector, step):
                    break
                step += 1
            else:

                f_current = target_func.subs(self.results[step])

                if abs(f_prev - f_current) > 0.000001:
                    f_prev = f_current
                    continue

            break

        print("Optimization complete. Final results are:")

        for k, v in self.results[step].items():
            print(k, "=", v)

        print("F =", f_current)

    def _part_vector(self, target_func, search_vector, step):

        subs_vector = {k: v for k, v in self.results[step].items() if k not in search_vector}

        objective = scipy_f_wrap(lambdify(search_vector, target_func.subs(subs_vector)))

        init_vector = [self.results[step][s] for s in search_vector]

        ieqcons_list = []
        eqcons_list = []
        COND = {}
        bounds_x = []

        for i in range(0, len(search_vector)):
            ieqcons_list.append(lambda x, i=i: x[i])
            COND[" >= 0 X%d" % i] = lambda x, i=i: x[i]
            bounds_x.append((0, 1))

        for j in range(1, self.tau + 1):

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

        for j in range(self.tau + 1, self.N + 1):

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

        min_vector = fmin_slsqp(func=objective,
                                x0=np.array(init_vector),
                                eqcons=eqcons_list,
                                ieqcons=ieqcons_list,
                                bounds=bounds_x,
                                full_output=True,
                                iter=1000,
                                acc=0.1,
                                epsilon=0.000001)

        if np.isnan(min_vector[1]):
            return False

        self.results[step + 1] = {k: v for k, v in self.results[step].items()}

        for i, s in enumerate(search_vector):
            self.results[step + 1].update({s: min_vector[0][i]})
            print(s, "=", min_vector[0][i])

        for name, cond in COND.items():
            print(cond(min_vector[0]), name)

        return True


if __name__ == "__main__":
    rs = RearmingSimulation()
    rs.init_equation_system()
    rs.find_initial_vector()
    rs.find_min_vector()
