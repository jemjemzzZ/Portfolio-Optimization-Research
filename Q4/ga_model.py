import random
import numpy as np
from deap import base, creator, tools, algorithms
import geatpy as ea

import tool

class GAModel:
    def __init__(self, historical_data, future_data, model_type):
        self.historical_data = historical_data.copy()
        self.future_data = future_data.copy()
        self.model_type = model_type
        self.unique_assets = historical_data.columns
        
    def evaluate(self, individual):
        selected_assets = [asset for asset, include in zip(self.unique_assets, individual) if include]
        
        if len(selected_assets) == 0:
            return -99999, -99999, 99999
        
        historical_data = self.historical_data[self.historical_data.columns[self.historical_data.columns.isin(selected_assets)]]
        future_data = self.future_data[self.future_data.columns[self.future_data.columns.isin(selected_assets)]]
        
        n = len(selected_assets)
        index_min_weight = [0 for _ in range(n)]
        index_max_weight = [1 for _ in range(n)]
        weight_constraints = list(zip(index_min_weight, index_max_weight))
        
        predict, _, _ = tool.evaluate(historical_data, future_data, weight_constraints, 'RB-SLSQP')
        expected_return, volatility, sharpe_ratio = predict
        
        return expected_return, sharpe_ratio, volatility
    
    def create_individual(self):
        individual = [random.randint(0, 1) for _ in range(len(self.unique_assets))]
        if sum(individual) == 0:
            individual[random.randint(0, len(self.unique_assets) - 1)] = 1
        return individual
    
    def main(self):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selNSGA2)
        
        def run_ga(pop_size, num_generations):
            pop = toolbox.population(n=pop_size)
            hof = tools.ParetoFront()
            logbook = tools.Logbook()
            stats = tools.Statistics(lambda ind: ind.fitness.values)

            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof,  verbose=True)
            
            return pop, hof, stats, logbook
        
        population, pareto_front, stats, logbook = run_ga(50, 10)

        pareto_front = [ind for ind in pareto_front if not all(x == 0 for x in ind)]
        def get_calculations(individual):
            selected_assets = [asset for asset, include in zip(self.unique_assets, individual) if include]
            
            historical_data = self.historical_data[self.historical_data.columns[self.historical_data.columns.isin(selected_assets)]]
            future_data = self.future_data[self.future_data.columns[self.future_data.columns.isin(selected_assets)]]
            
            n = len(selected_assets)
            index_min_weight = [0 for _ in range(n)]
            index_max_weight = [1 for _ in range(n)]
            weight_constraints = list(zip(index_min_weight, index_max_weight))
            
            predict, actual, weight = tool.evaluate(historical_data, future_data, weight_constraints, 'RB-SLSQP')
            
            return predict, actual, weight, selected_assets

        individual_predicts = {str(ind): get_calculations(ind)[0] for ind in pareto_front}
        individual_actuals = {str(ind): get_calculations(ind)[1] for ind in pareto_front}
        individual_weights = {str(ind): get_calculations(ind)[2] for ind in pareto_front}
        individual_selects = {str(ind): get_calculations(ind)[3] for ind in pareto_front}

        # Output
        for i, ind in enumerate(pareto_front):
            print(f"Pareto Front {i}")
            print(f"Selects: {individual_selects[str(ind)]}")
            print(f"Predicts: {individual_predicts[str(ind)]}")
            print(f"Actuals: {individual_actuals[str(ind)]}")
            print(f"Weights: {individual_weights[str(ind)]}")
            print("")
        
        return individual_predicts, individual_actuals, individual_weights, individual_selects


def evaluate(predicts, actuals, model_type):
    predicts = list(predicts.values())
    actuals = list(actuals.values())
    
    if model_type == 'RB-GA-Max-Ret':
        max_return = float('-inf')
        max_index = -1
        for i, (ret, vol, sharpe) in enumerate(predicts):
            if ret > max_return and ret != np.inf:
                max_return = ret
                max_index = i
        return predicts[max_index], actuals[max_index]
    elif model_type == 'RB-GA-Min-Vol':
        min_volatility = float('inf')
        min_index = -1
        for i, (ret, vol, sharpe) in enumerate(predicts):
            if vol < min_volatility:
                min_volatility = vol
                min_index = i
        return predicts[min_index], actuals[min_index]
    
    return None, None


"""
Version 2 with Geatpy
"""
class GAMultiProblem(ea.Problem):
    
    def __init__(self, historical_data, future_data, M=3):
        self.historical_data = historical_data.copy()
        self.future_data = future_data.copy()
        self.unique_assets = historical_data.columns
        
        name = 'GA multi'
        Dim = len(self.unique_assets)
        maxormins = [-1, -1, 1] # return, sharpe, vol
        varTypes = [1] * Dim
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        
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
            rows = Vars[i, :]
            w_sum[i] = np.sum(rows)
        
        return w_sum
        
    def evalVars(self, Vars):
        f_ret = np.zeros((Vars.shape[0], 1))
        f_sha = np.zeros((Vars.shape[0], 1))
        f_vol = np.zeros((Vars.shape[0], 1))
        
        f = np.ones((Vars.shape[0], 1))
        for i in range(Vars.shape[0]):
            rows = Vars[i, :]
            for j in range(len(rows)):
                individual = rows[j]
                
                selected_assets = [asset for asset, include in zip(self.unique_assets, individual) if include]
                historical_data = self.historical_data[self.historical_data.columns[self.historical_data.columns.isin(selected_assets)]]
                future_data = self.future_data[self.future_data.columns[self.future_data.columns.isin(selected_assets)]]
                n = len(selected_assets)
                index_min_weight = [0 for _ in range(n)]
                index_max_weight = [1 for _ in range(n)]
                weight_constraints = list(zip(index_min_weight, index_max_weight))
                predict, actual, weight = tool.evaluate(historical_data, future_data, weight_constraints, 'RB-SLSQP')
                
                f_ret[i], f_vol[i], f_sha[i] = predict
            
        CV = np.hstack([1 - self.constraint(Vars)])
        f = np.hstack([f_ret, f_sha, f_vol])
        
        return f, CV


def runGA(historical_data, future_data):
    problem = GAMultiProblem(historical_data, future_data)
    
    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='RI', NIND=50),
                                      MAXGEN=200,
                                      logTras=100)
    algorithm.mutOper.Pm = 0.2
    algorithm.recOper.XOVR = 0.9
    
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    
    print(res)