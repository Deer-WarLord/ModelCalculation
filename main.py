from math import exp
from sys import maxsize
import json
import leather
from pprint import pprint


class RearmingSimulation:
    def __init__(self):

        with open("initial_data.json") as json_file:
            initial_data = json_file.read()
            json_initial_data = json.loads(initial_data)

        self.C = float(json_initial_data["C"])
        self.L_old_0 = json_initial_data["L_old_0"]
        self.K_old_0 = [float(item) for item in json_initial_data["K_old_0"]]
        self.theta_old = [float(item) for item in json_initial_data["theta_old"]]
        self.S_phase_2 = [float(item) for item in json_initial_data["S_phase_2"]]
        self.S_phase_1 = [float(item) for item in json_initial_data["S_phase_1"]]
        self.S_tilda = [float(item) for item in json_initial_data["S_tilda"]]
        self.tau = json_initial_data["tau"]
        self.mu = [float(item) for item in json_initial_data["mu"]]
        self.nu = float(json_initial_data["nu"])
        self.L0 = json_initial_data["L0"]

        self.betta_old = [float(item) for item in json_initial_data["betta_old"]]
        self.alpha_old = [float(item) for item in json_initial_data["alpha_old"]]
        self.A_old = [float(item) for item in json_initial_data["A_old"]]

        self.betta_new = [float(item) for item in json_initial_data["betta_new"]]
        self.alpha_new = [float(item) for item in json_initial_data["alpha_new"]]
        self.A_new = [float(item) for item in json_initial_data["A_new"]]

        self.k_new = [float(item) for item in json_initial_data["k_new"]]

        self.a = [float(item) for item in json_initial_data["a"]]

        self.eps = float(json_initial_data["eps"])
        self.dh = float(json_initial_data["dh"])
        self.ds = int(1.0 / float(json_initial_data["ds"]))

        total = sum(self.S_phase_1) + sum(self.S_tilda)
        if total != 1:
            raise Exception("Share of the new investment in "
                            "total should be less then 1. Now %s" % total)

        total = sum(self.S_phase_2)
        if total != 1:
            raise Exception("Share of the old investment in "
                            "total should be less then 1. Now %s" % total)

        self.accumulations = {}
        self.K_new_0 = [0.0, 0.0, 0.0]
        self.L_new_0 = [0, 0, 0]
        self.Theta_new_prev = [0.0, 0.0, 0.0]
        self.sum_s_underline = 0.8
        self.sum_s_tilda = 0.2

    def xfrange(self, start, stop, step):
        i = 0
        while start + i * step < stop:
            yield start + i * step
            i += 1

    def capital_new(self, i, K_i, t, h, s_tilda_i, s_new_i, K1_new, L1_new):
        result = (-self.mu[i] * K_i + s_tilda_i * self.get_accumulation(t) +
                  s_new_i * self.Xi_new(1, K1_new, L1_new)) * h + K_i
        return 0 if result < 0 else result

    def captial_old(self, i, K_i, t, h, K1_old, L1_old, S_i):
        result = (-self.mu[i] * K_i + S_i * self.Xi_old(1, K1_old, L1_old)) * h + K_i
        return 0 if result < 0 else result

    def get_accumulation(self, t):
        return 0 if t / self.tau >= 2 else self.accumulations[t - self.tau]

    def Xi_old(self, i, K, L):
        return self.A_old[i] * pow(K, self.alpha_old[i]) * pow(L, self.betta_old[i])

    def Xi_new(self, i, K, L):
        return self.A_new[i] * pow(K, self.alpha_new[i]) * pow(L, self.betta_new[i])

    def L_new(self, i, K_new_i, L_new_prev):
        L = int(K_new_i / self.k_new[i])
        if (self.L_old_prev[i] + L_new_prev[i] - L) > 0:
            result = L
        else:
            result = self.L_old_prev[i] + L_new_prev[i]
        return result

    def Theta_new_i(self, L_new_i, L):
        return L_new_i / L

    def L_general(self, t):
        return self.L0 * exp(self.nu * t)

    def substract_theta(self, old_theta, new_theta, prev_new_theta):
        old = [0, 0, 0]
        for i in range(0, 3):
            theta = old_theta[i] - (new_theta[i] - prev_new_theta[i])
            if round(theta, 3) > 0:
                old[i] = theta
        return old

    def employee_condition(self, theta_old, theta_new):
        total = round(sum(theta_old) + sum(theta_new))
        return total == 1

    def X2_condition(self, X2_old, X2_new):
        return X2_old + X2_new >= self.C

    def matirial_condition(self, X, L):
        if L == 0.0:
            return False
        balance = round(X[0] / L - (X[0] / L * self.a[0] +
                                    X[1] / L * self.a[1] +
                                    X[2] / L * self.a[2]), 1)
        return balance == 0

    def matirial_difference(self, X, L):
        if L == 0.0:
            return self.eps + 1
        balance = round(X[0] / L - (X[0] / L * self.a[0] +
                                    X[1] / L * self.a[1] +
                                    X[2] / L * self.a[2]), 1)
        return abs(balance)

    def generate_s(self, size, share):
        for i in range(0, size, 1):
            for j in range(0, size, 1):
                for k in range(0, size, 1):
                    if (i + j + k) == size * share:
                        yield (i * 1.0 / size, j * 1.0 / size, k * 1.0 / size)

    def score_old(self, X):
        return X[1]

    def score(self, prev_vector, current_vector, diff):

        result = 0

        for i in [0, 1]:
            for j in [0, 1, 2]:
                result += abs(prev_vector[i][j] - current_vector[i][j])

        result += diff

        if diff == 0:
            result -= 1000

        return result

    def phase_1(self):

        K_old_prev = self.K_old_0
        L_old_prev = self.L_old_0

        result_vectors = {
            0: (K_old_prev, L_old_prev, 0, self.Xi_old(2, K_old_prev[2],
                                                       L_old_prev[2]), [0.4, 0.4, 0.0])
        }

        for t in self.xfrange(1, self.tau + 1, self.dh):

            is_found = False
            investment = self.Xi_old(1, K_old_prev[1], L_old_prev[1])
            self.accumulations[t] = investment
            L_old = [int(self.theta_old[i] * self.L_general(t)) for i in range(0, 3)]
            sum_L_old = sum(L_old)
            score_results = {}

            for S_phase_1 in self.generate_s(self.ds, self.sum_s_underline):

                K_old = [self.captial_old(i, K_old_prev[i], t, self.dh,
                                          K_old_prev[1], L_old_prev[1], S_phase_1[i]) for i in
                         range(0, 3)]

                X = [self.Xi_old(i, K_old[i], L_old[i]) for i in range(0, 3)]

                if self.matirial_condition(X, sum_L_old):
                    score_results[S_phase_1] = (self.score_old(X), K_old, X[2])
                    is_found = True

            if not is_found:
                raise Exception("The investments vector wasn't found on iteration %d" % t)

            print("Phase 1 iteration %.2f completed " % t)

            max_score = 0
            for s_vector, (score, K, X2) in score_results.items():
                if score > max_score:
                    result_vectors[t] = (K, L_old, investment, X2, s_vector)
                    max_score = score

            K_old_prev, L_old_prev = result_vectors[t][0], L_old

        grid = leather.Grid()

        chart = leather.Chart('Capital Phase 1')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][0][0], name="Sector 0")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][0][1], name="Sector 1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][0][2], name="Sector 2")
        grid.add_one(chart)

        chart = leather.Chart('Employees Phase 1')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][1][0], name="Sector 0")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][1][1], name="Sector 1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][1][2], name="Sector 2")
        grid.add_one(chart)

        chart = leather.Chart('X1_X2')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][2], name="X1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][3], name="X2")
        chart.add_line([(t, self.C) for t in range(0, self.tau + 1)],
                       x=lambda row, i: row[0], y=lambda row, i: row[1],
                       name="C")
        grid.add_one(chart)

        chart = leather.Chart('Share of the investments')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][4][0], name="Sector 0")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][4][1], name="Sector 1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][4][2], name="Sector 2")
        grid.add_one(chart)

        grid.to_svg('results_phase_1.svg')

        return K_old_prev

    def phase_2(self, K_old_prev):
        K_new_prev = self.K_new_0
        L_new_prev = self.L_new_0
        Theta_new_prev = self.Theta_new_prev

        Theta_old_prev = self.theta_old
        result_vectors = {}

        for t in self.xfrange(self.tau + 1, self.tau * 2 + 1, self.dh):

            score_results = {}
            current_state = {}
            K_new, L_new, Theta_old, K_old, L_old = ([], [], [], [], [])
            is_found = False
            self.L_old_prev = [int(Theta_old_prev[i] * self.L_general(t)) for i in range(0, 3)]
            total_L = sum(self.L_old_prev) + sum(L_new_prev)
            best_s_vector = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

            for S_tilda in self.generate_s(self.ds, self.sum_s_tilda):
                for S_new in self.generate_s(self.ds, 1):

                    K_new = [self.capital_new(i, K_new_prev[i], t, self.dh, S_tilda[i],
                                              S_new[i], K_new_prev[1], L_new_prev[1]) for i in
                             range(0, 3)]
                    L_new = [self.L_new(i, K_new[i], L_new_prev) for i in range(0, 3)]

                    Theta_new = [self.Theta_new_i(L_new[i], total_L) for i in range(0, 3)]
                    Theta_old = self.substract_theta(Theta_old_prev, Theta_new, Theta_new_prev)

                    if self.employee_condition(Theta_new, Theta_old):

                        L_old = [int(Theta_old[i] * total_L) for i in range(0, 3)]

                        K_old = [self.captial_old(i, K_old_prev[i], t, self.dh, K_old_prev[1],
                                                  self.L_old_prev[1], self.S_phase_2[i])
                                 for i in range(0, 3)]

                        X2_old = self.Xi_old(2, K_old[2], L_old[2])
                        X2_new = self.Xi_new(2, K_new[2], L_new[2])

                        if self.X2_condition(X2_old, X2_new):

                            X = [self.Xi_old(i, K_new[i], L_new[i]) for i in range(0, 3)]
                            diff = self.matirial_difference(X, sum(L_new))

                            if diff < self.eps:
                                score_results[S_new, S_tilda] = (self.score(best_s_vector,
                                                            (S_new, S_tilda), diff), Theta_new)
                                current_state[S_new, S_tilda] = (K_new, L_new, Theta_old, K_old,
                                                                 Theta_new, X, sum(L_new))
                                is_found = True

            if not is_found:
                pprint(Theta_old_prev)
                raise Exception("The investments vector wasn't found on iteration %d" % (t - self.tau))
            print("Was found %d suitable vectors" % len(current_state))

            min_score = maxsize
            best_s_vector = None
            for (s_new, s_tilda), (score, theta_vector) in score_results.items():
                if score < min_score:
                    result_vectors[t] = (s_new, theta_vector, s_tilda)
                    min_score = score
                    best_s_vector = s_new, s_tilda

            print("Phase 2 iteration %.2f completed " % t)

            if round(sum(Theta_old), 1) == 0:
                print("Re-arming was completed in %d year(s)" % (t - self.tau))
                break

            K_new_prev, L_new_prev, Theta_old_prev, K_old_prev, \
            Theta_new_prev, X, L = current_state[best_s_vector]

        pprint(Theta_old_prev)

        grid = leather.Grid()

        chart = leather.Chart('Share of the investments new')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][0][0], name="Sector 0")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][0][1], name="Sector 1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][0][2], name="Sector 2")
        grid.add_one(chart)

        chart = leather.Chart('Share of the investments tilda')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][2][0], name="Sector 0")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][2][1], name="Sector 1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][2][2], name="Sector 2")
        grid.add_one(chart)

        chart = leather.Chart('Employees')
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][1][0], name="Sector 0")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][1][1], name="Sector 1")
        chart.add_line(result_vectors.items(), x=lambda row, i: row[0],
                       y=lambda row, i: row[1][1][2], name="Sector 2")
        grid.add_one(chart)

        grid.to_svg('results_phase_2.svg')

    def simulate(self):
        self.phase_2(self.phase_1())


if __name__ == "__main__":
    rs = RearmingSimulation()
    rs.simulate()
