import random

# Objective function (Sphere function)
def objective_function(position):
    return sum(x**2 for x in position)

# Initialization function for the hive
def initialize_hive(population_size, dimension, bounds):
    return [[random.uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(population_size)]

# Function to simulate employed bees
def employed_bees(hive, trials, bounds):
    for i in range(len(hive)):
        new_bee = hive[i][:]
        k = random.randint(0, len(hive) - 1)
        j = random.randint(0, len(new_bee) - 1)
        phi = random.uniform(-1, 1)
        new_bee[j] = new_bee[j] + phi * (new_bee[j] - hive[k][j])
        new_bee[j] = min(max(new_bee[j], bounds[0]), bounds[1])

        if objective_function(new_bee) < objective_function(hive[i]):
            hive[i] = new_bee
            trials[i] = 0
        else:
            trials[i] += 1

# Function to simulate onlooker bees
def onlooker_bees(hive, trials, bounds):
    i = 0
    t = 0
    while t < len(hive):
        if random.random() < (objective_function(hive[i]) / sum(objective_function(bee) for bee in hive)):
            t += 1
            new_bee = hive[i][:]
            k = random.randint(0, len(hive) - 1)
            j = random.randint(0, len(new_bee) - 1)
            phi = random.uniform(-1, 1)
            new_bee[j] = new_bee[j] + phi * (new_bee[j] - hive[k][j])
            new_bee[j] = min(max(new_bee[j], bounds[0]), bounds[1])

            if objective_function(new_bee) < objective_function(hive[i]):
                hive[i] = new_bee
                trials[i] = 0
            else:
                trials[i] += 1
        i = (i + 1) % len(hive)

# Function to simulate scout bees
def scout_bees(hive, trials, bounds, limit):
    for i in range(len(trials)):
        if trials[i] > limit:
            hive[i] = [random.uniform(bounds[0], bounds[1]) for _ in range(len(hive[0]))]
            trials[i] = 0

# Main function to perform optimization
def abc_optimization(objective_function, bounds, dimension, population_size, max_generations, limit):
    hive = initialize_hive(population_size, dimension, bounds)
    trials = [0] * population_size

    for generation in range(max_generations):
        employed_bees(hive, trials, bounds)
        onlooker_bees(hive, trials, bounds)
        scout_bees(hive, trials, bounds, limit)
        best_solution = min(hive, key=objective_function)
        print(f"Generation {generation}: Best solution = {best_solution}, Value = {objective_function(best_solution)}")

    return min(hive, key=objective_function)

# Example usage
bounds = (-10, 10)
dimension = 5
population_size = 20
max_generations = 100
limit = 5
best_solution = abc_optimization(objective_function, bounds, dimension, population_size, max_generations, limit)
print(f"Final solution: {best_solution}, Value: {objective_function(best_solution)}")
