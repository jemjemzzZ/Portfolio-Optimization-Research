from deap import base, creator, tools, algorithms
import random, multiprocessing, time
from scoop import futures
import pickle

INDS_LEN = 100
NUM_LIMIT = 10
SUM_CONSTRAINT = 100
POP_SIZE = 50000
GEN_NUM = 200

step_size = 1 / (INDS_LEN / NUM_LIMIT)
block = [round(i * step_size, 3) for i in range(1, INDS_LEN // NUM_LIMIT+1)]
RESULT_LIST = block * NUM_LIMIT

def evaluate(individual):
    if not feasible(individual):
        repair(individual)
    
    r = 1
    min_len = min(len(individual), len(RESULT_LIST))
    for i in range(0, min_len):
        if individual[i] != 0:
            r *= individual[i] * RESULT_LIST[i]
    return r, 

def feasible(inidividual):
    f1 = sum(inidividual) == SUM_CONSTRAINT
    
    i = 0
    for ind in inidividual:
        if ind != 0:
            i += 1
    f2 = i == NUM_LIMIT
    
    return f1 and f2

def repair(individual):
    non_zero_indices = [i for i, x in enumerate(individual) if x > 0]
    zero_indices = [i for i, x in enumerate(individual) if x == 0]

    if len(non_zero_indices) > NUM_LIMIT:
        selected_indices = random.sample(non_zero_indices, NUM_LIMIT)
    else:
        selected_indices = non_zero_indices + random.sample(zero_indices, NUM_LIMIT - len(non_zero_indices))
    
    for i in range(len(individual)):
        if i not in selected_indices:
            individual[i] = 0
    
    if sum(individual) == 0:
        new_individual = create_individual()
        individual[:] = new_individual
    else:
        n = SUM_CONSTRAINT / sum(individual)
        for i in range(len(individual)):
            individual[i] = individual[i] * n

        d = SUM_CONSTRAINT - sum(individual)
        if selected_indices:
            chosen_index = random.choice(selected_indices)
            individual[chosen_index] += d

def create_individual():
    individual = [0] * INDS_LEN

    chosen_positions = random.sample(range(INDS_LEN), NUM_LIMIT)

    remaining = SUM_CONSTRAINT
    for i in range(NUM_LIMIT-1):
        individual[chosen_positions[i]] = random.randint(1, remaining - (NUM_LIMIT - (i + 1)))
        remaining -= individual[chosen_positions[i]]
    individual[chosen_positions[-1]] = remaining

    return individual


if __name__ == "__main__":
    start_time = time.time()
    
    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # toolbox.register("map", futures.map) # python -m scoop program.py
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Decorate evaluation function with feasibility check
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0, repair))

    # Evolutionary Algorithm
    def main():
        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", max)

        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GEN_NUM, stats=stats, halloffame=hof)

        return hof[0]
    
    decorated_evaluate = toolbox.evaluate
    try:
        pickled_evaluate = pickle.dumps(evaluate)
        print("The function is pickleable")
    except pickle.PicklingError as e:
        print("The function is not pickleable")
        print(e)

    # Run the GA
    # best_ind = main()
    
    # for i in range(0, len(best_ind), int(len(best_ind)/NUM_LIMIT)):
    #     print(best_ind[i:i + int(len(best_ind)/NUM_LIMIT)])
        
    # lst = [x for x in best_ind if x !=0]
    # print("Best Individual:", lst, "Fitness:", best_ind.fitness.values[0])
    
    # end_time = time.time()
    # duration = end_time - start_time
    # minutes = int(duration // 60)
    # seconds = int(duration % 60)
    # print(f"Total time: {minutes} minutes and {seconds} seconds")