import numpy as np

# Define the objective function to be optimized
def objective_function(x):
    return np.sin(x) + np.cos(2*x) + 0.5*x

# Genetic Algorithms (GA) algorithm
def ga_optimization(population_size, num_generations, mutation_rate, bounds):
    num_dimensions = len(bounds)
    population = np.random.uniform(bounds[0], bounds[1], size=(population_size, num_dimensions))

    for generation in range(num_generations):
        # Evaluate fitness of each individual in the population
        fitness = [objective_function(x) for x in population]

        # Select parents for mating based on fitness (roulette wheel selection)
        probabilities = 1 / np.array(fitness)
        probabilities /= np.sum(probabilities)
        parent_indices = np.random.choice(population_size, size=population_size, p=probabilities)

        # Perform crossover and mutation to create new offspring
        offspring = np.zeros_like(population)
        for i in range(population_size // 2):
            parent1, parent2 = population[parent_indices[i*2]], population[parent_indices[i*2+1]]
            crossover_point = np.random.randint(1, num_dimensions)
            offspring[i*2] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring[i*2+1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

            # Perform mutation
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(num_dimensions)
                offspring[i*2, mutation_point] = np.random.uniform(bounds[0], bounds[1])
                offspring[i*2+1, mutation_point] = np.random.uniform(bounds[0], bounds[1])

        # Replace old population with the new offspring
        population = offspring

    # Find the best individual in the final population
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    return best_solution, best_fitness

# Main function to run GA optimization
def main():
    population_size = 50
    num_generations = 100
    mutation_rate = 0.1
    bounds = (-2*np.pi, 2*np.pi)

    best_solution, best_fitness = ga_optimization(population_size, num_generations, mutation_rate, bounds)

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

if __name__ == "__main__":
    main()