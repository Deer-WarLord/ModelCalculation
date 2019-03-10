from test6 import RearmingSimulation

if __name__ == "__main__":
    rs = RearmingSimulation()
    rs.dt = 0.25
    rs.init_equation_system()
    rs.results_next = rs.load_pickle("tau2N4dt025")
    rs.find_min_vector(rs.results_next)
    rs.save_json(rs.results_next, "tau2N4dt025")
    rs.save_json(rs.labor, "labor_tau2N4dt025")