import json
import pickle
import logging
from itertools import chain
import numpy as np
from scipy.optimize import fmin_slsqp
from sympy import *


class Message(object):
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(**self.args[0]) if len(self.args) == 1 and isinstance(*self.args,
                                                                                     dict) else self.fmt.format(
            *self.args)


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super(StyleAdapter, self).__init__(logger, extra or {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, Message(msg, args), (), **kwargs)


DEBUG = False

log_level = logging.DEBUG if DEBUG else logging.INFO

log = logging.getLogger('rearming_simulation')
log.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - %(message)s', "%H:%M:%S")

ch = logging.StreamHandler()
ch.setLevel(log_level)
ch.setFormatter(formatter)
log.addHandler(ch)

fh = logging.FileHandler('results.log')
fh.setLevel(logging.DEBUG)
log.addHandler(fh)

log = StyleAdapter(log)

log.setLevel(log_level)


def scipy_f_wrap(f):
    return lambda x: np.array(f(*x))


class RearmingSimulation:
    def __init__(self):

        with open("initial_data_2.json") as json_file:
            initial_data = json_file.read()
            self.json_initial_data = json.loads(initial_data)

        self.C = float(self.json_initial_data["C"])
        self.ds = int(self.json_initial_data["ds"])
        # TODO Нужна проверка: шаг дискретизации dt не должен быть больше tau_ik
        self.tau = float(self.json_initial_data["tau"])
        self.tau_01 = float(self.json_initial_data["tau_01"])
        self.tau_02 = float(self.json_initial_data["tau_02"])
        self.tau_10 = float(self.json_initial_data["tau_10"])
        self.tau_12 = float(self.json_initial_data["tau_12"])
        self.tau_20 = float(self.json_initial_data["tau_20"])
        self.tau_21 = float(self.json_initial_data["tau_21"])
        self.N = self.tau * 2.0
        self.dt = float(self.json_initial_data["dh"])
        self.nu = float(self.json_initial_data["nu"])
        self.results = {0: {}}
        self.target_func = {}
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
    def generate_theta_psi(s, sh):
        # Не возможно генерировать theta psi на разных шагах т.к лаг может не совпадать
        # c шагом по дискретности и по размерности в любом случае будут всплывать провалы !!!
        def _inner_theta_psi(size, share):
            bound = size * share
            for i in range(0, size, 1):
                for j in range(0, size, 1):
                    for k in range(2, size, 1):
                        if (i + j + k) == bound:
                            yield (k * 1.0 / size, i * 1.0 / size, j * 1.0 / size)

        for theta_psi_0 in _inner_theta_psi(s, sh):
            for theta_psi_1 in _inner_theta_psi(s, sh):
                for theta_psi_2 in _inner_theta_psi(s, sh):
                    yield list(chain(theta_psi_0, theta_psi_1, theta_psi_2))

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

    def _build_capital_eq(self, j, i):
        self.EQ["K_old_{i}_{N}".format(N=j, i=i)] = (-self.EQ["mu_{i}".format(i=i)] *
                                                     self.EQ["K_old_{i}_{pN}".format(pN=j - self.dt, i=i)] +
                                                     self.EQ["su_old_{i}_{N}".format(N=j, i=i)] *
                                                     self.EQ["X_old_1_{pN}".format(pN=j - self.dt,
                                                                                   i=i)]) * self.dt + \
                                                    self.EQ["K_old_{i}_{pN}".format(pN=j - self.dt, i=i)]

    def _build_labor_eq(self, j, results, step, prefix="_old"):
        # noinspection PyCallingNonCallable
        self.labor[0].update({"L{}_{i}_{N}".format(prefix, N=j, i=i):
                                  str(self.EQ["L{}_{i}_{N}".format(prefix, N=j, i=i)].xreplace(results[step])) for i in
                              range(0, 3)})

    def _build_theta_eq(self, j, results, step, prefix="_new"):
        # noinspection PyCallingNonCallable
        self.labor[0].update({"theta{}_{i}_{N}".format(prefix, N=j, i=i):
                                  str(self.EQ["theta{}_{i}_{N}".format(prefix, N=j, i=i)].xreplace(results[step])) for i
                              in range(0, 3)})

    def _build_invest_eq(self, j, s, prefix):
        self.COND["invest{}_{N}".format(prefix, N=j)] = self.EQ["{0}{1}_0_{N}".format(s, prefix, N=j)] + \
                                                        self.EQ["{0}{1}_1_{N}".format(s, prefix, N=j)] + \
                                                        self.EQ["{0}{1}_2_{N}".format(s, prefix, N=j)]

        self.COND["invest{}_M_{N}".format(prefix, N=j)] = 1 - self.COND["invest{}_{N}".format(prefix, N=j)]  # >0

    def _build_balance_eq(self, j, b_prefix="", x_prefix="_old"):
        self.COND["balance{}_{N}".format(b_prefix, N=j)] = (self.EQ["X{0}_0_{N}".format(x_prefix, N=j)] -
                                                            (self.EQ["X{0}_0_{N}".format(x_prefix, N=j)] * self.EQ[
                                                                "a_0"] +
                                                             self.EQ["X{0}_1_{N}".format(x_prefix, N=j)] * self.EQ[
                                                                 "a_1"] +
                                                             self.EQ["X{0}_2_{N}".format(x_prefix, N=j)] * self.EQ[
                                                                 "a_2"])) / \
                                                           self.EQ["L_{N}".format(N=j)]

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

                self._build_capital_eq(j, i)

            self.EQ["L_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - self.dt)] * exp(self.nu * self.dt)

            # На второй фазе мы не можем использовать это уравнение т.к.
            # будет происходить переход ТС из старого сектора в новый.
            for i in range(0, 3):
                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ["L_{N}".format(N=j, i=i)] * \
                                                            self.EQ["theta_old_{i}".format(i=i)]

                # Тау не должно превышать длительность фазы накопления иначе нужно вводить отрицательный шаг
                # TODO сделать для этого правила проверку при инициализации
                for k in [kk for kk in range(0, 3) if kk != i]:
                    self.EQ["psi_{i}{k}_{N}".format(N=j, i=i, k=k)] = symbols("psi_{i}{k}_{N}".format(N=j, i=i, k=k))

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

            self._build_balance_eq(j)

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
            self.EQ["L_old_current_{i}_{pN}".format(pN=self.tau, i=i)] = self.EQ[
                "L_old_{i}_{N}".format(N=self.tau, i=i)]

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

            # Считаем общее кол-во ТС на данный момент
            self.EQ["L_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - self.dt)] * exp(self.nu * self.dt)

            # Считаем общий прирост ТС
            self.EQ["dL_{N}".format(N=j)] = self.EQ["L_{pN}".format(pN=j - self.dt)] - \
                                            self.EQ["L_{N}".format(N=j)]

            for i in range(0, 3):

                # Естественный прирост происходит в старые сектора экономики. Примем это отношение за константу.
                self.EQ["L_old_{i}_{N}".format(N=j, i=i)] = self.EQ[
                                                                "L_old_current_{i}_{pN}".format(pN=j - self.dt, i=i)] + \
                                                            self.EQ["dL_{N}".format(N=j)] * self.EQ[
                                                                "theta_old_{i}".format(i=i)]

                # Вектор распределения ТС из старого в новый сектор
                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)] = symbols("theta_new_{i}_{N}".format(N=j, i=i))

                # Переход части ТС в новый сектор
                self.EQ["L_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{pN}".format(pN=j - self.dt, i=i)] + \
                                                            self.EQ["theta_new_{i}_{N}".format(N=j, i=i)] * \
                                                            self.EQ["L_old_{i}_{N}".format(N=j, i=i)]

                # Уход ТС из старого сектора i в новый сектор i
                self.EQ["L_old_current_{i}_{N}".format(N=j, i=i)] = self.EQ["L_old_{i}_{N}".format(N=j, i=i)] * \
                                                                    (1 - self.EQ["theta_new_{i}_{N}".format(N=j, i=i)])

                self.EQ["sum_psi_{i}_{N}".format(N=j, i=i)] = 0

                for k in [kk for kk in range(0, 3) if kk != i]:
                    self.EQ["psi_{i}{k}_{N}".format(N=j, i=i, k=k)] = symbols("psi_{i}{k}_{N}".format(N=j, i=i, k=k))
                    self.EQ["sum_psi_{i}_{N}".format(N=j, i=i)] += self.EQ["psi_{i}{k}_{N}".format(N=j, i=i, k=k)]

                    # В новый сектор k из старого сектора i c лагом Tau из i в k
                    self.EQ["L_new_{k}{i}_{N}".format(N=j + getattr(self, "tau_{i}{k}".format(i=i, k=k)), i=i, k=k)] = \
                        self.EQ["L_old_{i}_{N}".format(N=j, i=i)] * \
                        self.EQ["psi_{i}{k}_{N}".format(N=j, i=i, k=k)]

                    # Уход ТС из старого сектора i в новый сектор k
                    self.EQ["L_old_current_{i}_{N}".format(N=j, i=i)] -= self.EQ["L_old_{i}_{N}".format(N=j, i=i)] * \
                                                                         self.EQ["psi_{i}{k}_{N}".format(N=j, i=i, k=k)]

                    # В новый сектор i из старого сектора k на текущем щаге с уже учтенным лагом Tau
                    # Условие if hasattr не работает !!! Нужно реализовывать через исключение
                    try:
                        self.EQ["L_new_{i}_{N}".format(N=j, i=i)] = self.EQ["L_new_{i}_{N}".format(N=j, i=i)] + \
                                                                    self.EQ["L_new_{i}{k}_{N}".format(N=j, i=i, k=k)]
                    except KeyError as e:
                        pass

                # for searching optimal vector using algorithm we still need conditions for Labor and Psi
                self.COND["theta_psi_bound_{i}_{N}".format(i=i, N=j)] = 1 - (self.EQ["sum_psi_{i}_{N}".format(N=j, i=i)] +
                                                                self.EQ["theta_new_{i}_{N}".format(N=j, i=i)]) # >= 0

            self.COND["L_balance_{N}".format(N=j)] = self.EQ["L_{N}".format(N=j)] - \
                                                     (self.EQ["L_new_0_{N}".format(N=j)] +
                                                      self.EQ["L_new_1_{N}".format(N=j)] +
                                                      self.EQ["L_new_2_{N}".format(N=j)])  # >0

            for i in range(0, 3):
                self.EQ["X_new_{i}_{N}".format(N=j, i=i)] = self.EQ["A_new_{i}".format(i=i)] * \
                                                            self.EQ["L_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_new_{i}".format(i=i)] * \
                                                            self.EQ["K_new_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_new_{i}".format(i=i)]

            for i in range(0, 3):
                self.EQ["su_old_{i}_{N}".format(N=j, i=i)] = symbols("su_old_{i}_{N}".format(N=j, i=i))
                self._build_capital_eq(j, i)

            for i in range(0, 3):
                self.EQ["X_old_{i}_{N}".format(N=j, i=i)] = self.EQ["A_old_{i}".format(i=i)] * \
                                                            self.EQ["L_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["alpha_old_{i}".format(i=i)] * \
                                                            self.EQ["K_old_{i}_{N}".format(N=j, i=i)] ** \
                                                            self.EQ["betta_old_{i}".format(i=i)]

            self._build_invest_eq(j, "st", "_new")

            self._build_invest_eq(j, "su", "_old")

            self._build_balance_eq(j, "_new", "_new")

            self.COND["consuming_bound_{N}".format(N=j)] = self.EQ["X_new_2_{N}".format(N=j)] + \
                                                           self.EQ["X_old_2_{N}".format(N=j)]

            self.COND["consuming_bound_L_{N}".format(N=j)] = self.COND["consuming_bound_{N}".format(N=j)] - self.C  # >0

    def find_initial_vector(self):

        log.info("Phase 1 is started")

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
                log.info("nothing was found")
                break

            values_3 = dict([(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)])
            values_2 = dict([(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])

            for k in self.xfrange(j + self.dt, self.tau + self.dt, self.dt):
                self.EQ["K_old_0_{N}".format(N=k)] = self.EQ["K_old_0_{N}".format(N=k)].xreplace(values_3)
                self.EQ["K_old_1_{N}".format(N=k)] = self.EQ["K_old_1_{N}".format(N=k)].xreplace(values_3)
                self.EQ["K_old_2_{N}".format(N=k)] = self.EQ["K_old_2_{N}".format(N=k)].xreplace(values_3)

                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].xreplace(values_2)

                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].xreplace(values_3)

            log.info("step {} s: {}", j, s)

            self.results[0].update({self.EQ["su_old_{i}_{N}".format(N=j, i=i)]: s[i] for i in range(0, 3)})

        log.info("Phase 1 is completed")
        log.info("Phase 2 is started")

        is_complete = True

        prev_theta = [0.5, 0.5, 0.5]

        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):
            _st_new, _st_old, _su_old, _theta_psi = None, None, None, None

            K_new_0_subs = self.EQ["K_new_0_{N}".format(N=j)].xreplace(self.results[0])
            K_new_0 = lambdify([self.EQ["st_old_0_{N}".format(N=j - self.tau)], self.EQ["st_new_0_{N}".format(N=j)]],
                               K_new_0_subs)

            K_new_1_subs = self.EQ["K_new_1_{N}".format(N=j)].xreplace(self.results[0])
            K_new_1 = lambdify([self.EQ["st_old_1_{N}".format(N=j - self.tau)], self.EQ["st_new_1_{N}".format(N=j)]],
                               K_new_1_subs)

            K_new_2_subs = self.EQ["K_new_2_{N}".format(N=j)].xreplace(self.results[0])
            K_new_2 = lambdify([self.EQ["st_old_2_{N}".format(N=j - self.tau)], self.EQ["st_new_2_{N}".format(N=j)]],
                               K_new_2_subs)

            L_balance_subs = self.COND["L_balance_{N}".format(N=j)].xreplace(self.results[0])

            # Может происходить разрыв из-за того что предыдущий шаг учитывается при подсчетете текущего
            # ТС в старом секторе, а лаг вливание в новый сектор не всегда совпадает с размером шага дискретизации.
            # Предыдущий шаг будет заполняться словарем results так же как и лаг т.к он может быть либо больше либо
            # равен шагу дискретизации
            # TODO специфика формата управления  не дает нам перелить все ресурсы из первого сектора почему ???
            L_balance = lambdify([self.EQ["theta_new_0_{N}".format(N=j)],
                                  self.EQ["theta_new_1_{N}".format(N=j)],
                                  self.EQ["theta_new_2_{N}".format(N=j)]], L_balance_subs)

            consumption_subs = self.COND["consuming_bound_{N}".format(N=j)].xreplace(self.results[0])

            consumption = lambdify((self.EQ["st_new_0_{N}".format(N=j)],
                                    self.EQ["st_new_1_{N}".format(N=j)],
                                    self.EQ["st_new_2_{N}".format(N=j)],
                                    self.EQ["su_old_2_{N}".format(N=j)],
                                    self.EQ["theta_new_2_{N}".format(N=j)],
                                    self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_2_{N}".format(N=j - self.tau)]), consumption_subs)

            balance_subs = self.COND["balance_new_{N}".format(N=j)].xreplace(self.results[0])
            balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                self.EQ["st_new_1_{N}".format(N=j)],
                                self.EQ["st_new_2_{N}".format(N=j)],
                                self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_2_{N}".format(N=j - self.tau)],
                                self.EQ["theta_new_0_{N}".format(N=j)],
                                self.EQ["theta_new_1_{N}".format(N=j)],
                                self.EQ["theta_new_2_{N}".format(N=j)]], balance_subs)

            for st_new in self.generate_s(int(self.ds / 4), 1.0):

                for theta_psi in self.generate_theta_psi(10, 0.5):

                    for st_old in self.generate_s(self.ds, 0.3):

                        if K_new_0(st_old[0], st_new[0]) >= 0 and \
                                        K_new_1(st_old[1], st_new[1]) >= 0 and \
                                        K_new_2(st_old[2], st_new[2]) >= 0 and \
                                        L_balance(*chain([theta_psi[0], theta_psi[3], theta_psi[6]])) >= 0:

                            b = balance(*chain(st_new, st_old, [theta_psi[0], theta_psi[3], theta_psi[6]]))

                            if abs(b) < 5:
                                log.debug("b = {: 0.5f} st_old = {} st_new = {} theta_psi = {}", b, st_old, st_new,
                                          theta_psi)

                            if -1.0 <= round(b, 1) <= 1.0:
                                _st_new, _st_old, _theta_psi = st_new, st_old, theta_psi
                                max_c = -1
                                for su_old in self.generate_s(self.ds, 1.0):
                                    c = consumption(*chain(st_new, [su_old[2]], [theta_psi[6]], st_old))
                                    if c >= self.C:
                                        _su_old = su_old
                                        break
                                    if c >= max_c:
                                        max_c = c

                                log.info("b = {: 0.5f} max_c = {: 0.1f} st_old = {} st_new = {} theta_psi = {}",
                                         b, max_c, st_old, st_new, theta_psi)

                        if _su_old:
                            break

                    if _su_old:
                        break

                if _su_old:
                    break

            if not _su_old:
                log.info("nothing was found")
                is_complete = False
                break

            log.info("step {} st_new: {} su_old: {} st_old: {} theta_psi: {}", j, _st_new, _su_old, _st_old, _theta_psi)

            for i in range(0, 3):
                self.results[0].update({self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)]: _st_old[i]})
                self.results[0].update({self.EQ["su_old_{i}_{N}".format(i=i, N=j)]: _su_old[i]})
                self.results[0].update({self.EQ["st_new_{i}_{N}".format(i=i, N=j)]: _st_new[i]})
                self.results[0].update({self.EQ["theta_new_{i}_{N}".format(i=i, N=j)]: _theta_psi[i * 3]})
                prev_theta[i] = _theta_psi[i * 3]

            self.results[0].update({self.EQ["psi_01_{N}".format(N=j)]: _theta_psi[1]})
            self.results[0].update({self.EQ["psi_02_{N}".format(N=j)]: _theta_psi[2]})

            self.results[0].update({self.EQ["psi_10_{N}".format(N=j)]: _theta_psi[4]})
            self.results[0].update({self.EQ["psi_12_{N}".format(N=j)]: _theta_psi[5]})

            self.results[0].update({self.EQ["psi_20_{N}".format(N=j)]: _theta_psi[7]})
            self.results[0].update({self.EQ["psi_21_{N}".format(N=j)]: _theta_psi[8]})

            l_old = [self.EQ["L_old_{i}_{N}".format(N=j, i=i)].xreplace(self.results[0]) for i in range(0, 3)]
            l_new = [self.EQ["L_new_{i}_{N}".format(N=j, i=i)].xreplace(self.results[0]) for i in range(0, 3)]

            log.info("L_old: {} L_new: {}", l_old, l_new)

            target_func = self.EQ["X_new_0_{N}".format(N=j)] / self.EQ["L_new_0_{N}".format(N=j)] + \
                          self.EQ["X_new_1_{N}".format(N=j)] / self.EQ["L_new_1_{N}".format(N=j)] + \
                          self.EQ["X_new_2_{N}".format(N=j)] / self.EQ["L_new_2_{N}".format(N=j)]

            log.info("F = {}", target_func.xreplace(self.results[0]))
            log.info("dL = {}", [l_old[i] - l_new[i] for i in range(0, 3)])

        self.results[0].update({self.EQ["psi_01_{N}".format(N=self.N)]: 0.0})
        self.results[0].update({self.EQ["psi_02_{N}".format(N=self.N)]: 0.0})

        self.results[0].update({self.EQ["psi_10_{N}".format(N=self.N)]: 0.0})
        self.results[0].update({self.EQ["psi_12_{N}".format(N=self.N)]: 0.0})

        self.results[0].update({self.EQ["psi_20_{N}".format(N=self.N)]: 0.0})
        self.results[0].update({self.EQ["psi_21_{N}".format(N=self.N)]: 0.0})

        log.info("Phase 2 is completed")
        return is_complete

    def find_initial_vector_using_prev(self, prev_dt, prev_tau, prev_N):
        log.info("Phase 1 is started")

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
                            log.debug("f_min = {}", f_min)
                            if f_min <= 1:
                                break
            if not s:
                log.info("nothing was found")
                is_complete = False
                break

            values_3 = dict([(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(0, 3)])
            values_2 = dict([(self.EQ["su_old_{i}_{N}".format(N=j, i=i)], s[i]) for i in range(1, 3)])

            for k in self.xfrange(j + self.dt, self.tau + self.dt, self.dt):
                self.EQ["K_old_0_{N}".format(N=k)] = self.EQ["K_old_0_{N}".format(N=k)].xreplace(values_3)
                self.EQ["K_old_1_{N}".format(N=k)] = self.EQ["K_old_1_{N}".format(N=k)].xreplace(values_3)
                self.EQ["K_old_2_{N}".format(N=k)] = self.EQ["K_old_2_{N}".format(N=k)].xreplace(values_3)

                self.EQ["X_old_2_{N}".format(N=k)] = self.EQ["X_old_2_{N}".format(N=k)].xreplace(values_2)

                self.COND["balance_{N}".format(N=k)] = self.COND["balance_{N}".format(N=k)].xreplace(values_3)

            log.info("step {} s: {}", j, s)

            self.results_next[0].update({self.EQ["su_old_{i}_{N}".format(N=j, i=i)]: s[i] for i in range(0, 3)})

            s_state[j] = round(sum(s), 2)

        if not is_complete:
            log.info("Phase 1 isn't completed")
            return

        log.info("Phase 1 is completed")
        log.info("Phase 2 is started")

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

            K_new_0_subs = self.EQ["K_new_0_{N}".format(N=j)].xreplace(self.results_next[0])
            K_new_0 = lambdify([self.EQ["st_old_0_{N}".format(N=j - self.tau)], self.EQ["st_new_0_{N}".format(N=j)]],
                               K_new_0_subs)

            K_new_1_subs = self.EQ["K_new_1_{N}".format(N=j)].xreplace(self.results_next[0])
            K_new_1 = lambdify([self.EQ["st_old_1_{N}".format(N=j - self.tau)], self.EQ["st_new_1_{N}".format(N=j)]],
                               K_new_1_subs)

            K_new_2_subs = self.EQ["K_new_2_{N}".format(N=j)].xreplace(self.results_next[0])
            K_new_2 = lambdify([self.EQ["st_old_2_{N}".format(N=j - self.tau)], self.EQ["st_new_2_{N}".format(N=j)]],
                               K_new_2_subs)

            L_balance_subs = self.COND["L_balance_{N}".format(N=j)].xreplace(self.results_next[0])

            L_balance = lambdify([self.EQ["theta_new_0_{N}".format(N=j)],
                                  self.EQ["psi_01_{N}".format(N=j - getattr(self, "tau_01"))],
                                  self.EQ["psi_02_{N}".format(N=j - getattr(self, "tau_02"))],
                                  self.EQ["theta_new_1_{N}".format(N=j)],
                                  self.EQ["psi_10_{N}".format(N=j - getattr(self, "tau_10"))],
                                  self.EQ["psi_12_{N}".format(N=j - getattr(self, "tau_12"))],
                                  self.EQ["theta_new_2_{N}".format(N=j)],
                                  self.EQ["psi_20_{N}".format(N=j - getattr(self, "tau_20"))],
                                  self.EQ["psi_21_{N}".format(N=j - getattr(self, "tau_21"))]], L_balance_subs)

            consumption_subs = self.COND["consuming_bound_{N}".format(N=j)].xreplace(self.results_next[0])

            consumption = lambdify((self.EQ["st_new_0_{N}".format(N=j)],
                                    self.EQ["st_new_1_{N}".format(N=j)],
                                    self.EQ["st_new_2_{N}".format(N=j)],
                                    self.EQ["su_old_2_{N}".format(N=j)],
                                    self.EQ["theta_new_2_{N}".format(N=j)],
                                    self.EQ["psi_20_{N}".format(N=j - getattr(self, "tau_20"))],
                                    self.EQ["psi_21_{N}".format(N=j - getattr(self, "tau_21"))],
                                    self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                    self.EQ["st_old_2_{N}".format(N=j - self.tau)]), consumption_subs)

            balance_subs = self.COND["balance_new_{N}".format(N=j)].xreplace(self.results_next[0])
            balance = lambdify([self.EQ["st_new_0_{N}".format(N=j)],
                                self.EQ["st_new_1_{N}".format(N=j)],
                                self.EQ["st_new_2_{N}".format(N=j)],
                                self.EQ["st_old_0_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_1_{N}".format(N=j - self.tau)],
                                self.EQ["st_old_2_{N}".format(N=j - self.tau)],
                                self.EQ["theta_new_0_{N}".format(N=j)],
                                self.EQ["psi_01_{N}".format(N=j - getattr(self, "tau_01"))],
                                self.EQ["psi_02_{N}".format(N=j - getattr(self, "tau_02"))],
                                self.EQ["theta_new_1_{N}".format(N=j)],
                                self.EQ["psi_10_{N}".format(N=j - getattr(self, "tau_10"))],
                                self.EQ["psi_12_{N}".format(N=j - getattr(self, "tau_12"))],
                                self.EQ["theta_new_2_{N}".format(N=j)],
                                self.EQ["psi_20_{N}".format(N=j - getattr(self, "tau_20"))],
                                self.EQ["psi_21_{N}".format(N=j - getattr(self, "tau_21"))]], balance_subs)

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
            # TODO create theta_psi generator around !!!

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
                                        log.debug("f_min = {}", f_min)
                                        if f_min <= 1:
                                            break

            if not _su_old:
                log.info("nothing was found")
                is_complete = False
                break

            log.info("step {} st_new: {} su_old: {} st_old: {}", j, _st_new, _su_old, _st_old)

            for i in range(0, 3):
                self.results_next[0].update({self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)]: _st_old[i]})
                self.results_next[0].update({self.EQ["su_old_{i}_{N}".format(i=i, N=j)]: _su_old[i]})
                self.results_next[0].update({self.EQ["st_new_{i}_{N}".format(i=i, N=j)]: _st_new[i]})

            l_old = [self.EQ["L_old_{i}_{N}".format(N=j, i=i)].xreplace(self.results_next[0]) for i in range(0, 3)]
            l_new = [self.EQ["L_new_{i}_{N}".format(N=j, i=i)].xreplace(self.results_next[0]) for i in range(0, 3)]

            log.info("L_old: {} L_new: {}", l_old, l_new)

            target_func = self.EQ["X_new_0_{N}".format(N=j)] / self.EQ["L_new_0_{N}".format(N=j)] + \
                          self.EQ["X_new_1_{N}".format(N=j)] / self.EQ["L_new_1_{N}".format(N=j)] + \
                          self.EQ["X_new_2_{N}".format(N=j)] / self.EQ["L_new_2_{N}".format(N=j)]

            log.info("F = {}", target_func.xreplace(self.results_next[0]))
            log.info("dL = {}", [l_old[i] - l_new[i] for i in range(0, 3)])

        log.info("Phase 2 is completed")
        return is_complete

    def find_min_vector(self, results):

        log.info("Start building target_func for minimization")

        target_func = -(self.EQ["X_new_0_{N}".format(N=self.N)] / self.EQ["L_new_0_{N}".format(N=self.N)] +
                        self.EQ["X_new_1_{N}".format(N=self.N)] / self.EQ["L_new_1_{N}".format(N=self.N)] +
                        self.EQ["X_new_2_{N}".format(N=self.N)] / self.EQ["L_new_2_{N}".format(N=self.N)])

        log.debug("Inited target_func")

        step = 0
        f_prev = -target_func.xreplace(results[0])

        log.debug("Subs target_func")

        f_current = 0
        self.labor = {}
        self.capital = {}

        log.debug("Start minimization iterations")

        while True:

            for j in reversed(list(self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt))):

                log.debug("step = {step} Minimize st_new_{N} su_old_{N} su_old_{pN} st_old_{pN} theta_psi_{N}",
                          {"step": step, "N": j, "pN": j - self.tau})

                search_vector = [self.EQ["st_new_{i}_{N}".format(i=i, N=j)] for i in range(0, 3)] + \
                                [self.EQ["su_old_{i}_{N}".format(i=i, N=j)] for i in range(0, 3)] + \
                                [self.EQ["su_old_{i}_{N}".format(i=i, N=j - self.tau)] for i in range(0, 3)] + \
                                [self.EQ["st_old_{i}_{N}".format(i=i, N=j - self.tau)] for i in range(0, 3)] + \
                                [self.EQ["theta_new_0_{N}".format(N=j)],
                                 self.EQ["psi_01_{N}".format(N=j)],
                                 self.EQ["psi_02_{N}".format(N=j)],
                                 self.EQ["theta_new_1_{N}".format(N=j)],
                                 self.EQ["psi_10_{N}".format(N=j)],
                                 self.EQ["psi_12_{N}".format(N=j)],
                                 self.EQ["theta_new_2_{N}".format(N=j)],
                                 self.EQ["psi_20_{N}".format(N=j)],
                                 self.EQ["psi_21_{N}".format(N=j)]]

                if not self._part_vector(target_func, search_vector, step, results):
                    break
                step += 1
            else:

                f_current = -target_func.xreplace(results[step])
                log.debug("f_prev = {} f_current = {}", f_prev, f_current)
                delta = f_prev - f_current
                log.debug("Delta = {}", delta)
                if abs(delta) > 0.001:
                    f_prev = f_current
                    log.debug("step = {} Go to another one minimization cycle ", step)
                    continue
                elif delta > 0:
                    log.debug("Optimization on step {} made function worse, return to previous results", step)
                    step -= 1

            break

        log.info("Optimization complete. Final results are:")

        for k, v in results[step].items():
            log.debug("{} = {}", k, v)

        target_func_val = target_func.xreplace(results[step])
        log.info("F = {}", -target_func_val)

        l_old = [self.EQ["L_old_{i}_{N}".format(N=self.N, i=i)].xreplace(self.results[step]) for i in range(0, 3)]
        l_new = [self.EQ["L_new_{i}_{N}".format(N=self.N, i=i)].xreplace(self.results[step]) for i in range(0, 3)]

        log.info("L_old: {} L_new: {}", l_old, l_new)
        # log.info("dL = {}", [l_old[i] - l_new[i] for i in range(0, 3)])

        self.target_func.update({self.dt: target_func_val})

        self.labor[0] = {}
        self.capital[0] = {}
        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):
            self.labor[0].update({"L_{N}".format(N=j): str(self.EQ["L_{N}".format(N=j)])})

            self._build_labor_eq(j, results, step, "_new")
            self._build_labor_eq(j, results, step, "_old")

            self._build_theta_eq(j, results, step, "_new")
            # self._build_theta_eq(j, results, step, "_old")

            self.capital[0].update({"K_new_{i}_{N}".format(N=j, i=i):
                                        str(self.EQ["K_new_{i}_{N}".format(N=j, i=i)].xreplace(results[step])) for i in
                                    range(0, 3)})

    def _part_vector(self, target_func, search_vector, step, results):

        subs_vector = {k: v for k, v in results[step].items() if k not in search_vector}

        log.debug("Start lambdifing objective")

        objective = scipy_f_wrap(lambdify(search_vector, target_func.xreplace(subs_vector)))

        log.debug("Finish lambdifing objective")

        init_vector = [results[step][s] for s in search_vector]

        ieqcons_list = []
        eqcons_list = []
        COND = {}
        bounds_x = []

        log.debug("Build X >= 0 conditions")

        for i in range(0, len(search_vector)):
            ieqcons_list.append(lambda x, i=i: x[i])
            COND[" >= 0 X%d" % i] = lambda x, i=i: x[i]
            bounds_x.append((0, 1))

        log.debug("Build 1th phase conditions")

        for j in self.xfrange(self.dt, self.tau + self.dt, self.dt):

            cond_list = (
                ("== 0 invest_M_{N}".format(N=j), self.COND["invest_M_{N}".format(N=j)].xreplace(subs_vector)),
                ("== 0 balance_{N}".format(N=j), self.COND["balance_{N}".format(N=j)].xreplace(subs_vector)),
            )

            for name, cond in cond_list:
                if len(cond.free_symbols) > 0:
                    f = scipy_f_wrap(lambdify(search_vector, cond))
                    eqcons_list.append(f)
                    COND[name] = f

            cond = self.COND["consuming_bound_L_{N}".format(N=j)].xreplace(subs_vector)

            if len(cond.free_symbols) > 0:
                f = scipy_f_wrap(lambdify(search_vector, cond))
                ieqcons_list.append(f)
                COND[" >= 0 consuming_bound_L_{N}".format(N=j)] = f

        log.debug("Build 2th phase conditions")

        for j in self.xfrange(self.tau + self.dt, self.N + self.dt, self.dt):

            log.debug("j = {} Build invest and balance cond", j)

            cond_list = (
                ("== 0 invest_old_M_{N}".format(N=j), self.COND["invest_old_M_{N}".format(N=j)].xreplace(subs_vector)),
                ("== 0 invest_new_M_{N}".format(N=j), self.COND["invest_new_M_{N}".format(N=j)].xreplace(subs_vector)),
                ("== 0 balance_new_{N}".format(N=j), self.COND["balance_new_{N}".format(N=j)].xreplace(subs_vector))
            )

            log.debug("j = {} Finish subs invest and balance cond", j)

            for name, cond in cond_list:
                if len(cond.free_symbols) > 0:
                    f = scipy_f_wrap(lambdify(search_vector, cond))
                    eqcons_list.append(f)
                    COND[name] = f

            log.debug("j = {} Build consuming bound cond", j)

            cond = self.COND["consuming_bound_L_{N}".format(N=j)].xreplace(subs_vector)

            log.debug("j = {} Finish subs consuming bound cond", j)

            if len(cond.free_symbols) > 0:
                f = scipy_f_wrap(lambdify(search_vector, cond))
                ieqcons_list.append(f)
                COND[" >= 0 consuming_bound_L_{N}".format(N=j)] = f

            log.debug("j = {} Build labor balance cond", j)

            cond = self.COND["L_balance_{N}".format(N=j)].xreplace(subs_vector)

            log.debug("j = {} Finish subs labor balance cond", j)

            if len(cond.free_symbols) > 0:
                f = scipy_f_wrap(lambdify(search_vector, cond))
                ieqcons_list.append(f)
                COND[" >= 0 L_balance_{N}".format(N=j)] = f

            for i in range(0, 3):

                log.debug("j = {} Theta-Psi-{} bound cond", j, i)

                cond = self.COND["theta_psi_bound_{i}_{N}".format(i=i, N=j)].xreplace(subs_vector)

                log.debug("j = {} Finish subs theta-Psi-{} bound cond", j, i)

                if len(cond.free_symbols) > 0:
                    f = scipy_f_wrap(lambdify(search_vector, cond))
                    ieqcons_list.append(f)
                    COND[" >= 0 theta_psi_bound_{i}_{N}".format(N=j, i=i)] = f

        log.debug("Run fmin_slsqp")

        min_vector = fmin_slsqp(func=objective,
                                x0=np.array(init_vector),
                                eqcons=eqcons_list,
                                ieqcons=ieqcons_list,
                                bounds=bounds_x,
                                iter=1000,
                                acc=0.1)

        if np.isnan(min_vector[0]):
            log.debug("fmin_slsqp returned Nan results")
            return False

        log.debug("fmin_slsqp returned results")
        results[step + 1] = {k: v for k, v in results[step].items()}

        for i, s in enumerate(search_vector):
            results[step + 1].update({s: min_vector[i]})
            log.debug("{} = {}", s, min_vector[i])

        for name, cond in COND.items():
            log.debug("{} {}", cond(min_vector), name)

        return True

    @staticmethod
    def save_pickle(results, f_name):
        with open('%s_v2.pickle' % f_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved result vector to file %s" % f_name)

    @staticmethod
    def save_json(results, fname):
        with open('%s_v2.json' % fname, 'w') as handle:
            json.dump({str(k): {str(nk): nv for nk, nv in v.items()}
                       for k, v in results.items()}, handle, ensure_ascii=False)
        log.info("Saved result vector to file %s" % fname)

    @staticmethod
    def save_target_f_json(results, f_name):
        with open('%s_v2.json' % f_name, 'w') as handle:
            json.dump({str(k): str(v) for k, v in results.items()}, handle, ensure_ascii=False)
        log.info("Saved target function values to file %s" % f_name)

    @staticmethod
    def load_pickle(f_name):
        with open('%s_v2.pickle' % f_name, 'rb') as handle:
            results = pickle.load(handle)
            log.info("Loaded result vector from file")
        return results


def save_data(rs, tau, N, dt):
    rs.save_pickle(rs.results, "tau{}N{}dt{}".format(tau, N, dt))
    rs.save_json(rs.results, "tau{}N{}dt{}".format(tau, N, dt))
    rs.save_json(rs.labor, "labor_tau{}N{}dt{}".format(tau, N, dt))
    rs.save_json(rs.capital, "capital_tau{}N{}dt{}".format(tau, N, dt))
    rs.save_target_f_json(rs.target_func, "f_tau{}N{}dt{}".format(tau, N, dt))


if __name__ == "__main__":
    rs = RearmingSimulation()

    # rs.dt = 1.0
    # rs.init_equation_system()
    # if rs.find_initial_vector():
    #     rs.find_min_vector(rs.results)
    #     save_data(rs, 2, 4, 1)

    # rs.dt = 0.5
    # rs.init_equation_system()
    # if rs.find_initial_vector():
    #     rs.find_min_vector(rs.results)
    #     save_data(rs, 2, 4, "05")

    rs.dt = 0.25
    rs.init_equation_system()
    if rs.find_initial_vector():
        rs.find_min_vector(rs.results)
        save_data(rs, 2, 4, "025")
