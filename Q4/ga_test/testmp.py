"""
Using multi-processing to improve the performance of GA from DEAP package.
"""


from deap import base, creator, tools, algorithms
import random, multiprocessing, time


INDS_LEN = 2000
NUM_LIMIT = 10
SUM_CONSTRAINT = 100
POP_SIZE = 5000
GEN_NUM = 100

step_size = 1 / (INDS_LEN / NUM_LIMIT)
block = [round(i * step_size, 3) for i in range(1, INDS_LEN // NUM_LIMIT+1)]
RESULT_LIST = block * NUM_LIMIT


def evaluate(individual):
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
        return new_individual
    
    else:
        n = SUM_CONSTRAINT / sum(individual)
        for i in range(len(individual)):
            individual[i] = int(individual[i] * n)

        d = SUM_CONSTRAINT - sum(individual)
        if selected_indices:
            chosen_index = random.choice(selected_indices)
            individual[chosen_index] += d
    
    return individual


def create_individual():
    individual = [0] * INDS_LEN

    chosen_positions = random.sample(range(INDS_LEN), NUM_LIMIT)

    remaining = SUM_CONSTRAINT
    for i in range(NUM_LIMIT-1):
        individual[chosen_positions[i]] = random.randint(1, remaining - (NUM_LIMIT - (i + 1)))
        remaining -= individual[chosen_positions[i]]
    individual[chosen_positions[-1]] = remaining

    return individual


def eaSimpleMod(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        # fix ind
        for i in range(len(offspring)):
            if not feasible(offspring[i]):
                offspring[i] = repair(offspring[i])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


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

    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)

    # Evolutionary Algorithm
    def main():
        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", max)

        eaSimpleMod(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GEN_NUM, stats=stats, halloffame=hof)

        return hof[0]

    # Run the GA
    best_ind = main()

    for i in range(0, len(best_ind), int(len(best_ind)/NUM_LIMIT)):
        print(best_ind[i:i + int(len(best_ind)/NUM_LIMIT)])
        
    lst = [x for x in best_ind if x !=0]
    print("Best Individual:", lst, "Fitness:", best_ind.fitness.values[0])

    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"Total time: {minutes} minutes and {seconds} seconds")
    
    pool.close()
    pool.join()