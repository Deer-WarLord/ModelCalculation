import json
from itertools import chain
from time import gmtime, strftime

from sympy import *

EPS = 0.001


class RearmingSimulation:
    def __init__(self):
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

        with open("initial_data.json") as json_file:
            initial_data = json_file.read()
            json_initial_data = json.loads(initial_data)

        # TODO make substitution on Lambda
        # TODO find min Lambda
        # TODO make external cycle while difference between F_min >= EPS

        tau = json_initial_data["tau"]
        self.N = tau * 2
        dt = float(json_initial_data["dh"])
        nu = float(json_initial_data["nu"])
        self.ds = int(1.0 / float(json_initial_data["ds"]))
        self.C = float(json_initial_data["C"])
        self.EQ["L_0"] = json_initial_data["L0"]
        self.EQ["a"] = [float(item) for item in json_initial_data["a"]]

        for i in range(0, 3):
            self.EQ["mu_{i}".format(i=i)] = float(json_initial_data["mu"][i])
            self.EQ["K_old_{i}_0".format(i=i)] = float(json_initial_data["K_old_0"][i])
            self.EQ["L_old_{i}_0".format(i=i)] = float(json_initial_data["L_old_0"][i])
            self.EQ["theta_old_{i}".format(i=i)] = float(json_initial_data["theta_old"][i])
            self.EQ["A_old_{i}".format(i=i)] = float(json_initial_data["A_old"][i])
            self.EQ["alpha_old_{i}".format(i=i)] = float(json_initial_data["alpha_old"][i])
            self.EQ["betta_old_{i}".format(i=i)] = float(json_initial_data["betta_old"][i])
            self.EQ["a_{i}".format(i=i)] = float(json_initial_data["a"][i])

        self.EQ["X_old_1_0"] = self.EQ["A_old_1"] * self.EQ["L_old_1_0"] ** self.EQ["alpha_old_1"] * \
                               self.EQ["K_old_1_0"] ** self.EQ["betta_old_1"]

        for j in range(1, tau):
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
                                                        self.EQ["X_old_1_{pN}".format(pN=j-1, i=i)]

            self.COND["invest_{N}".format(N=j)] = sum([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] +
                                                       self.EQ["st_old_{i}_{N}".format(N=j, i=i)] for i in
                                                       range(0, 3)]) - 1

            self.COND["balance_{N}".format(N=j)] = (self.EQ["X_old_0_{N}".format(N=j)] -
                                                   (self.EQ["X_old_0_{N}".format(N=j)] * self.EQ["a_0"] +
                                                    self.EQ["X_old_1_{N}".format(N=j)] * self.EQ["a_1"] +
                                                    self.EQ["X_old_2_{N}".format(N=j)] * self.EQ["a_2"])) / \
                                                   self.EQ["L_{N}".format(N=j)]

        for j in range(1, tau):
            s = None
            consumption = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["X_old_2_{N}".format(N=j)])
            balance = lambdify([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] for i in range(0, 3)],
                               self.COND["balance_{N}".format(N=j)])
            for S_phase_1 in self.generate_s(self.ds, 0.9):
                if consumption(S_phase_1[2]) >= self.C and -1.0 <= balance(*S_phase_1) <= 1.0:
                    s = S_phase_1
                    break
            if not s:
                print("nothing was found")
                break
            values = [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)]
            for k in range(j + 1, tau):
                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].subs(
                    [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])
                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].subs(values)
            print("step {j} s: {s}".format(j=j, s=s))
            self.results[0].update({self.EQ["su_old_{i}_{N}".format(N=j, i=i)]: s[i] for i in range(0, 3)})

        for i in range(0, 3):
            self.EQ["theta_old_{i}_{pN}".format(i=i, pN=tau - 1)] = self.EQ["theta_old_{i}".format(i=i)]
            self.EQ["K_new_{i}_{pN}".format(pN=tau - 1, i=i)] = 0.0
            self.EQ["I_{i}_{tau}".format(tau=tau, i=i)] = 0.0
            self.EQ["A_new_{i}".format(i=i)] = float(json_initial_data["A_new"][i])
            self.EQ["alpha_new_{i}".format(i=i)] = float(json_initial_data["alpha_new"][i])
            self.EQ["betta_new_{i}".format(i=i)] = float(json_initial_data["betta_new"][i])
            self.EQ["k_{i}".format(i=i)] = float(json_initial_data["k_new"][i])

        self.EQ["X_new_1_{pN}".format(pN=tau - 1)] = 0.0

        for j in range(tau, self.N):
            print(strftime("%H:%M:%S", gmtime()), "st_new K_new", j)
            for i in range(0, 3):
                self.EQ["st_new_{i}_{N}".format(N=j, i=i)] = symbols("st_new_{i}_{N}".format(N=j, i=i), negative=False)

                self.EQ["K_new_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_new_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                             self.EQ["I_{i}_{pN}".format(pN=j + 1 - tau, i=i)] +
                                                             self.EQ["st_new_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_new_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                            self.EQ["K_new_{i}_{pN}".format(pN=j - 1, i=i)]

            self.EQ["L_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - 1)] * exp(nu * dt)

            for i in range(0, 3):
                self.EQ["L_new_v1_{i}_{N}".format(N=j, i=i)] = self.EQ["K_new_{i}_{N}".format(N=j, i=i)] / \
                                                               self.EQ["k_{i}".format(i=i)]

                self.EQ["L_new_v2_{i}_{N}".format(N=j, i=i)] = self.EQ["theta_old_{i}_{pN}".format(i=i, pN=j - 1)] * \
                                                               self.EQ["L_{N}".format(N=j)]

                self.EQ["L_new_{i}_{N}".format(N=j, i=i)] = Min(self.EQ["L_new_v1_{i}_{N}".format(N=j, i=i)],
                                                                self.EQ["L_new_v2_{i}_{N}".format(N=j, i=i)])

            print(strftime("%H:%M:%S", gmtime()), "L_new", j)

            for i in range(0, 3):
                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{N}".format(N=j, i=i)] / \
                                                                self.EQ["L_{N}".format(N=j, i=i)]

                self.EQ["X_new_{i}_{N}".format(N=j, i=i)] = self.EQ["A_new_{i}".format(i=i)] * \
                                                            self.EQ["L_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_new_{i}".format(i=i)] * \
                                                            self.EQ["K_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_new_{i}".format(i=i)]

            print(strftime("%H:%M:%S", gmtime()), "theta_new X_new", j)

            for i in range(0, 3):
                self.EQ["theta_old_{i}_{N}".format(i=i, N=j)] = self.EQ["theta_old_{i}_{pN}".format(i=i, pN=j - 1)] - \
                                                                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)]

                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ["theta_old_{i}_{N}".format(i=i, N=j)] * \
                                                            self.EQ["L_{N}".format(N=j, i=i)]

            print(strftime("%H:%M:%S", gmtime()), "theta_old L_old", j)

            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i))
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                             self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                             self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                             self.EQ["X_old_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                            self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)]

            print(strftime("%H:%M:%S", gmtime()), "su_old K_old", j)

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * \
                                                            self.EQ["L_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_old_{i}".format(i=i)] * \
                                                            self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_old_{i}".format(i=i)]

            print(strftime("%H:%M:%S", gmtime()), "X_old", j)

            self.COND["balance_new_{N}".format(N=j)] = (self.EQ["X_new_0_{N}".format(N=j)] -
                                                       (self.EQ["X_new_0_{N}".format(N=j)] * self.EQ["a_0"] +
                                                        self.EQ["X_new_1_{N}".format(N=j)] * self.EQ["a_1"] +
                                                        self.EQ["X_new_2_{N}".format(N=j)] * self.EQ["a_2"])) / \
                                                       self.EQ["L_{N}".format(N=j)]

            print(strftime("%H:%M:%S", gmtime()), "balance_new", j)

            self.COND["consuming_bound_{N}".format(N=j)] = self.EQ["X_new_2_{N}".format(N=j)] + \
                                                           self.EQ["X_old_2_{N}".format(N=j)]

            print(strftime("%H:%M:%S", gmtime()), "consuming_bound", j)

        for j in range(tau, self.N):
            _st_new, _st_old, _su_old = None, None, None
            consumption_subs = self.COND["consuming_bound_{N}".format(N=j)].subs(self.results[0])
            consumption = lambdify((self.EQ["st_new_2_{N}".format(N=j)],
                                    self.EQ["su_old_2_{N}".format(N=j)],
                                    self.EQ["st_old_2_{tau}".format(tau=j + 1 - tau)]), consumption_subs)

            balance_subs = self.COND["balance_new_{N}".format(N=j)].subs(self.results[0])
            balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                self.EQ["st_new_1_{N}".format(N=j)],
                                self.EQ["st_new_2_{N}".format(N=j)],
                                self.EQ["st_old_0_{N}".format(N=j + 1 - tau)],
                                self.EQ["st_old_1_{N}".format(N=j + 1 - tau)],
                                self.EQ["st_old_2_{N}".format(N=j + 1 - tau)]], balance_subs)

            b_prev = 0

            for st_new in self.generate_s(self.ds, 1.0):

                for st_old in self.generate_s(self.ds, 0.1):

                    b = balance(*chain(st_new, st_old))

                    if int(b_prev) != int(b) and -5.0 < b < 5.0:
                        print("step", j, st_old, b)

                    if -1.0 <= b <= 1.0:
                        _st_new, _st_old = st_new, st_old
                        b_prev = b

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
                self.results[0].update({self.EQ["st_old_{i}_{N}".format(i=i, N=j + 1 - tau)]: _st_old[i]})
                self.results[0].update({self.EQ["su_old_{i}_{N}".format(i=i, N=j)]: _su_old[i]})
                self.results[0].update({self.EQ["st_new_{i}_{N}".format(i=i, N=j)]: _st_new[i]})

            l_old = [self.EQ["L_old_{i}_{N}".format(N=j, i=i)].subs(self.results[0]) for i in range(0, 3)]
            l_new = [self.EQ["L_new_{i}_{N}".format(N=j, i=i)].subs(self.results[0]) for i in range(0, 3)]

            print("L_old: {l_old} L_new: {l_new}".format(l_old=l_old, l_new=l_new))

            for k in range(j + 1, tau):
                self.COND["consuming_bound_{N}".format(N=k)] = \
                    self.COND["consuming_bound_{N}".format(N=k)].subs(self.results[0])
                self.COND["balance_new_{N}".format(N=k)] = \
                    self.COND["balance_new_{N}".format(N=k)].subs(self.results[0])

            target_func = sum([EPS * (self.EQ["theta_old_{i}".format(i=i)] -
                                      self.EQ["theta_new_{i}_{N}".format(N=j, i=i)]) ** 2 for i in range(0, 3)])

            print(strftime("%H:%M:%S", gmtime()), "F =", target_func.subs(self.results[0]))


if __name__ == "__main__":
    rs = RearmingSimulation()
    rs.init_equation_system()
