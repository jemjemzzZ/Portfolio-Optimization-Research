import numpy as np
import geatpy as ea

import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

from deap import base, creator, tools, algorithms
import random, time


class RBwithGA(ea.Problem):
    
    def __init__(self, cov, risk_alloc, bounds):
        # evaluation setup
        self.cov = cov
        self.risk_alloc = risk_alloc
        self.bounds = bounds
        
        # ea problem setup
        name = 'Risk Budget with Evolutionary Algorithm'
        M = 1
        maxormins = [1]
        Dim = len(cov)
        varTypes = [0] * Dim
        lb = [i[0] for i in bounds]
        ub = [i[1] for i in bounds]
        lbin = [1] * Dim
        ubin = [1] * Dim
        
        # ea initialisation
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)
    
    def constraint(self, Vars):
        w_sum = np.zeros((Vars.shape[0], 1))
        
        for i in range(Vars.shape[0]):
            w_sum[i] = np.sum(Vars[i, :])
        
        return w_sum
    
    def risk_budget(self, rows):
        cov_matrix = np.array(self.cov)
        
        def volatility(weights):
            weights = np.array(weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return vol

        def risk_contribution(weights):
            weights = np.array(weights)
            vol = volatility(weights)
            mrc = np.dot(cov_matrix, weights) / vol
            trc = np.multiply(mrc, weights)
            return trc

        def risk_parity(weights):
            vol = volatility(weights)
            risk_target_pct = np.array(self.risk_alloc)
            risk_target = np.multiply(vol, risk_target_pct)
            trc = risk_contribution(weights)
            J = np.sqrt(sum(np.square(trc - risk_target)))
            return J
        
        return risk_parity(rows)
    
    def evalVars(self, Vars):
        f = np.zeros((Vars.shape[0], 1))
        
        for i in range(Vars.shape[0]):
            rows = Vars[i, :]
            f[i] = self.risk_budget(rows)
        
        CV = np.hstack([np.abs(self.constraint(Vars) - 1)])
        return f, CV



def runRBGA(cov, risk_alloc, bounds):
    problem = RBwithGA(cov, risk_alloc, bounds)
    
    algorithm = ea.soea_SEGA_templet(problem,
                                     ea.Population(Encoding='RI', NIND=5000),
                                     MAXGEN=3000,
                                     logTras=100)
    algorithm.mutOper.F = 0.7
    algorithm.recOper.XOVR = 0.7
    
    res = ea.optimize(algorithm,
                      verbose=False,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    
    # print(res)
    return res['Vars'], res['ObjV']



def runRBGAwithDEAP(cov, risk_alloc):
    num_assets = len(cov)
    
    def evaluate(individual):
        if not feasible(individual):
            repair(individual)
        
        cov_matrix = np.array(cov)
        
        def volatility(weights):
            weights = np.array(weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return vol

        def risk_contribution(weights):
            weights = np.array(weights)
            vol = volatility(weights)
            mrc = np.dot(cov_matrix, weights) / vol
            trc = np.multiply(mrc, weights)
            return trc

        def risk_parity(weights):
            vol = volatility(weights)
            risk_target_pct = np.array(risk_alloc)
            risk_target = np.multiply(vol, risk_target_pct)
            trc = risk_contribution(weights)
            J = np.sqrt(sum(np.square(trc - risk_target)))
            return J
        
        return risk_parity(individual), 
    
    def feasible(individual):
        return sum(individual) == 1 and all(0 <= x <= 1 for x in individual)
    
    def repair(individual):
        total = sum(individual)
        if total != 0:
            individual[:] = [x / total for x in individual]
        else:
            individual[:] = [1/len(individual) for _ in len(individual)]
        return individual
    
    def mutUniformFloat(individual, low, up, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low, up)
        return individual,
    
    # Setup DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutUniformFloat, low=0, up=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def main():
        pop = toolbox.population(n=500)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", min)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000000, stats=stats, halloffame=hof)
        return hof[0]

    start_time = time.time()
    best_ind = main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("Best Individual:", best_ind, "Fitness:", best_ind.fitness.values[0])
    
    return best_ind, best_ind.fitness.values[0]



def runRBSLSQP(cov, risk_alloc, bounds):
    cov_matrix = np.array(cov)
    n = len(cov_matrix)
    x0 = [1 / n for _ in range(n)]

    def volatility(weights):
        weights = np.array(weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return vol

    def risk_contribution(weights):
        weights = np.array(weights)
        vol = volatility(weights)
        mrc = np.dot(cov_matrix, weights) / vol
        trc = np.multiply(mrc, weights)
        return trc

    def risk_parity(weights):
        vol = volatility(weights)
        risk_target_pct = np.array(risk_alloc)
        risk_target = np.multiply(vol, risk_target_pct)
        trc = risk_contribution(weights)
        J = np.sqrt(sum(np.square(trc - risk_target)))
        return J

    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    solution = minimize(risk_parity, x0=x0,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    if solution.success:
        final_weights = solution.x
    else:
        final_weights = [1 / n for _ in range(n)]
    return final_weights, risk_parity(final_weights)




if __name__ == "__main__":
    num_assets = 20
    random_matrix = np.random.rand(num_assets, num_assets)
    cov = np.dot(random_matrix, random_matrix.T)
    cov = (cov + cov.T) / 2
    np.fill_diagonal(cov, 1)
    
    # cov = [[1.        , 3.80756876, 2.92526191, 2.60845318, 2.82623432, 3.16068328, 2.9428879 , 3.1404206 , 3.32079592, 3.09109874],
    #     [3.80756876, 1.        , 2.20733146, 2.60406106, 2.20126318, 2.62349067, 2.04465151, 2.74352817, 2.50654272, 2.76995942],
    #     [2.92526191, 2.20733146, 1.        , 1.29154096, 1.66607752, 1.94542491, 2.00129904, 1.65327867, 2.06070693, 1.73164543],
    #     [2.60845318, 2.60406106, 1.29154096, 1.        , 1.58950688, 1.86019043, 1.37043545, 2.92217508, 2.43925871, 2.54505221],
    #     [2.82623432, 2.20126318, 1.66607752, 1.58950688, 1.        , 1.48886934, 1.94467454, 2.2865428 , 1.87593328, 1.98976173],
    #     [3.16068328, 2.62349067, 1.94542491, 1.86019043, 1.48886934, 1.        , 1.64930312, 2.06627995, 2.85955454, 1.76301283],
    #     [2.9428879 , 2.04465151, 2.00129904, 1.37043545, 1.94467454, 1.64930312, 1.        , 2.14187044, 2.28837051, 1.50530465],
    #     [3.1404206 , 2.74352817, 1.65327867, 2.92217508, 2.2865428 , 2.06627995, 2.14187044, 1.        , 2.70284828, 2.60642259],
    #     [3.32079592, 2.50654272, 2.06070693, 2.43925871, 1.87593328, 2.85955454, 2.28837051, 2.70284828, 1.        , 1.83033621],
    #     [3.09109874, 2.76995942, 1.73164543, 2.54505221, 1.98976173, 1.76301283, 1.50530465, 2.60642259, 1.83033621, 1.        ]]

    risk_alloc = np.random.rand(num_assets)
    risk_alloc /= risk_alloc.sum()
    
    # risk_alloc = [0.07160934, 0.12251125, 0.15482847, 0.09486425, 0.10247162, 0.07068814, 0.19496963, 0.01467748, 0.13639808, 0.03698173]
    
    index_min_weight = [0 for _ in range(num_assets)]
    index_max_weight = [1 for _ in range(num_assets)]
    bounds = list(zip(index_min_weight, index_max_weight))
    
    
    print("=======================TEST ONE=======================")
    weights_one, performance_one = runRBSLSQP(cov, risk_alloc, bounds)
    print(weights_one)
    print(performance_one)
    print("=======================TEST ONE=======================")
    
    print("=======================TEST TWO=======================")
    weights_two, performance_two = runRBGA(cov, risk_alloc, bounds)
    print(weights_two)
    print(performance_two)
    print("=======================TEST TWO=======================")
    
    # print("=======================TEST Three=======================")
    # weights_three, performance_three = runRBGAwithDEAP(cov, risk_alloc)
    # print(weights_three)
    # print(performance_three)
    # print("=======================TEST Three=======================")