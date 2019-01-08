import json
from sympy import *
from scipy.optimize import minimize_scalar
from math import fabs as math_fabs, sqrt as math_sqrt

EPS = 0.001

class RearmingSimulation:
    # def __init__(self):
    # 
    #     self.S_phase_2 = [float(item) for item in json_initial_data["S_phase_2"]]
    #     self.S_phase_1 = [float(item) for item in json_initial_data["S_phase_1"]]
    #     self.S_tilda = [float(item) for item in json_initial_data["S_tilda"]]
    #
    #     self.eps = float(json_initial_data["eps"])
    # 
    #     total = sum(self.S_phase_1) + sum(self.S_tilda)
    #     if total != 1:
    #         raise Exception("Share of the new investment in "
    #                         "total should be less then 1. Now %s" % total)
    # 
    #     total = sum(self.S_phase_2)
    #     if total != 1:
    #         raise Exception("Share of the old investment in "
    #                         "total should be less then 1. Now %s" % total)
    # 
    # 
    #     self.Theta_new_prev = [0.0, 0.0, 0.0]
    #     self.sum_s_underline = 0.8
    #     self.sum_s_tilda = 0.2

    def generate_s(self, size, share):
        for i in range(0, size, 1):
            for j in range(0, size, 1):
                for k in range(0, size, 1):
                    if (i + j + k) == size * share:
                        yield (i * 1.0 / size, j * 1.0 / size, k * 1.0 / size)

    def init_equation_system(self):

        with open("initial_data.json") as json_file:
            initial_data = json_file.read()
            json_initial_data = json.loads(initial_data)

        # TODO make substitution on Lambda
        # TODO find min Lambda
        # TODO make external cycle while difference between F_min >= EPS
        self.EQ = {}
        self.COND = {}
        self.res0 = {}

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

        self.EQ["X_old_1_0"] = self.EQ["A_old_1"] * self.EQ["L_old_1_0"] ** self.EQ["alpha_old_1"] * self.EQ["K_old_1_0"] ** self.EQ["betta_old_1"]

        for j in range(1, tau):
            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i), negative=False)
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] * self.EQ[
                    "K_old_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                        self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                        self.EQ["X_old_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                       self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)]

                # print("K_old_{i}_{N} =".format(N=j, i=i), self.EQ["K_old_{i}_{N}".format(N=j, i=i)])

            self.EQ["L_{N}".format(N=j, i=i)] = self.EQ["L_{pN}".format(pN=j - 1, i=i)] * exp(nu * dt)
            # print("L_{N} =".format(N=j, i=i), self.EQ["L_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ["L_{N}".format(N=j, i=i)] * self.EQ["theta_old_{i}".format(i=i)]
                # print("L_old_{i}_{N} =".format(N=j, i=i), self.EQ["L_old_{i}_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * self.EQ["L_old_{i}_{N}".format(
                    N=j, i=i)] ** self.EQ["alpha_old_{i}".format(i=i)] * self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** self.EQ[
                    "betta_old_{i}".format(i=i)]

                # print("X_old_{i}_{N} =".format(N=j, i=i), self.EQ["X_old_{i}_{N}".format(N=j, i=i)])
                self.EQ["st_old_{i}_{N}".format(N=j, i=i)] = symbols("st_old_{i}_{N}".format(N=j, i=i), negative=False)

            for i in range(0, 3):
                self.EQ["I_{i}_{N}".format(N=j, i=i)] = self.EQ["st_old_{i}_{N}".format(N=j, i=i)] * self.EQ[
                    "X_old_1_{N}".format(N=j, i=i)]
                # print("I_{i}_{N} =".format(N=j, i=i), self.EQ["I_{i}_{N}".format(N=j, i=i)])

            self.COND["invest_{N}".format(N=j)] = sum([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] +
                                                   self.EQ["st_old_{i}_{N}".format(N=j, i=i)] for i in range(0, 3)]) - 1

            # print("invest_{N} =".format(N=j), self.COND["invest_{N}".format(N=j)])

            self.COND["balance_{N}".format(N=j)] = Abs(self.EQ["X_old_0_{N}".format(N=j)] -
                sum([self.EQ["X_old_{i}_{N}".format(N=j, i=i)] * self.EQ["a_{i}".format(i=i)] for i in range(0, 3)])) / self.EQ["L_{N}".format(N=j, i=i)]

            # print("balance_{N} =".format(N=j), self.COND["balance_{N}".format(N=j)])

            # print("-"*200+"\n")
        # ConditionSet()

        for j in range(1, tau):
            s = None
            consumption = lambdify(self.EQ["su_old_2_{N}".format(N=j)], self.EQ["X_old_2_{N}".format(N=j)])
            balance = lambdify([self.EQ["su_old_{i}_{N}".format(N=j, i=i)] for i in range(0, 3)], self.COND["balance_{N}".format(N=j)])
            for S_phase_1 in self.generate_s(self.ds, 0.8):
                if consumption(S_phase_1[2]) >= self.C and balance(*S_phase_1) <= 1:
                    s = S_phase_1
                    break
            if not s:
                print("nothing was found")
                break
            values = [(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)]
            for k in range(j+1, tau):
                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].subs([(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])
                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].subs(values)
            print("step {j} s: {s}".format(j=j, s=s))

        return


        # --------------------------------------------------------------

        for i in range(0, 3):
            self.EQ["K_new_{i}_{pN}".format(pN=tau - 1, i=i)] = 0.0
            self.EQ["I_{i}_{tau}".format(tau=tau, i=i)] = 0.0
            self.EQ["A_new_{i}".format(i=i)] = float(json_initial_data["A_new"][i])
            self.EQ["alpha_new_{i}".format(i=i)] = float(json_initial_data["alpha_new"][i])
            self.EQ["betta_new_{i}".format(i=i)] = float(json_initial_data["betta_new"][i])
            self.EQ["k_{i}".format(i=i)] = float(json_initial_data["k_new"][i])

        self.EQ["X_new_1_{pN}".format(pN=tau - 1)] = 0.0

        for j in range(tau, self.N):
            for i in range(0, 3):
                self.EQ["st_new_{i}_{N}".format(N=j, i=i)] = symbols("st_new_{i}_{N}".format(N=j, i=i))

                self.EQ["K_new_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] * self.EQ[
                    "K_new_{i}_{pN}".format(pN=j - 1, i=i)] + self.EQ["I_{i}_{pN}".format(pN=j + 1 - tau, i=i)] +
                                                        self.EQ["st_new_{i}_{N}".format(N=j, i=i)] * self.EQ[
                                                            "X_new_1_{pN}".format(pN=j - 1, i=i)]) * dt + self.EQ[
                                                           "K_new_{i}_{pN}".format(pN=j - 1, i=i)]

                # print("K_new_{i}_{N} =".format(N=j, i=i), self.EQ["K_new_{i}_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["L_new_{i}_{N}".format(N=j, i=i)] = self.EQ["K_new_{i}_{N}".format(N=j, i=i)] / self.EQ["k_{i}".format(i=i)]
                # print("L_new_{i}_{N} =".format(N=j, i=i), self.EQ["L_new_{i}_{N}".format(N=j, i=i)])

            self.EQ["L_{N}".format(N=j, i=i)] = self.EQ["L_{pN}".format(pN=j - 1, i=i)] * exp(nu * dt)
            # print("L_{N} =".format(N=j, i=i), self.EQ["L_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{N}".format(N=j, i=i)] / self.EQ[
                    "L_{N}".format(N=j, i=i)]

                self.EQ["X_new_{i}_{N}".format(N=j, i=i)] = self.EQ["A_new_{i}".format(i=i)] * self.EQ["L_new_{i}_{N}".format(
                    N=j, i=i)] ** self.EQ["alpha_new_{i}".format(i=i)] * self.EQ["K_new_{i}_{N}".format(N=j, i=i)] ** self.EQ[
                    "betta_new_{i}".format(i=i)]

                # print("X_new_{i}_{N} =".format(N=j, i=i), self.EQ["X_new_{i}_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = (self.EQ["theta_old_{i}".format(i=i)] - self.EQ[
                    "theta_new_{i}_{N}".format(N=j, i=i)]) * \
                                                       self.EQ["L_{N}".format(N=j, i=i)]
                # print("L_old_{i}_{N} =".format(N=j, i=i), self.EQ["L_old_{i}_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i))
                self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] * self.EQ[
                    "K_old_{i}_{pN}".format(pN=j - 1, i=i)] +
                                                        self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                        self.EQ["X_old_1_{pN}".format(pN=j - 1, i=i)]) * dt + \
                                                       self.EQ["K_old_{i}_{pN}".format(pN=j - 1, i=i)]
                # print("K_old_{i}_{N} =".format(N=j, i=i), self.EQ["K_old_{i}_{N}".format(N=j, i=i)])

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * self.EQ["L_old_{i}_{N}".format(
                    N=j, i=i)] ** self.EQ["alpha_old_{i}".format(i=i)] * self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** self.EQ[
                    "betta_old_{i}".format(i=i)]
                # print("X_old_{i}_{N} =".format(N=j, i=i), self.EQ["X_old_{i}_{N}".format(N=j, i=i)])

                # print("-"*200+"\n")

        self.target_func = sum([EPS * (self.EQ["theta_old_{i}".format(i=i)] - self.EQ["theta_new_{i}_{N}".format(N=self.N-1, i=i)]) ** 2 for i in range(0,3)])

        print("F = ", self.target_func)

        for s in sorted(self.target_func.free_symbols, key=lambda x: str(x)[::-1]):
            print(s)

        consuming_bound_new = lambda X2_new, X2_old, C: X2_new + X2_old >= C

        investments_balance_old = lambda su: math_fabs(sum(su) - 1) <= EPS

        investments_balance_new = lambda st: math_fabs(sum(st) - 1) <= EPS


if __name__ == "__main__":
    rs = RearmingSimulation()
    rs.init_equation_system()
