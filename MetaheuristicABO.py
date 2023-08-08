import numpy as np

# Define the objective function to be optimized
def objective_function(x):
    return (x - 2) ** 2

# Artificial Bee Colony (ABC) algorithm
def abc_optimization(num_bees, num_iterations, num_trials, bounds):
    num_dimensions = len(bounds)
    best_solution = None
    best_fitness = float('inf')

    for _ in range(num_iterations):
        # Employed bees phase
        employed_bees = np.random.uniform(bounds[0], bounds[1], size=(num_bees, num_dimensions))
        employed_fitness = [objective_function(x) for x in employed_bees]

        # Onlooker bees phase
        onlooker_probabilities = 1 / np.array(employed_fitness)
        onlooker_probabilities /= np.sum(onlooker_probabilities)

        for i in range(num_bees):
            selected_bee = np.random.choice(num_bees, p=onlooker_probabilities)
            trial_solution = employed_bees[i] + np.random.uniform(-1, 1, size=num_dimensions) * (employed_bees[i] - employed_bees[selected_bee])
            trial_fitness = objective_function(trial_solution)

            if trial_fitness < employed_fitness[i]:
                employed_bees[i] = trial_solution
                employed_fitness[i] = trial_fitness

        # Scout bees phase
        best_bee_index = np.argmin(employed_fitness)
        if employed_fitness[best_bee_index] < best_fitness:
            best_solution = employed_bees[best_bee_index]
            best_fitness = employed_fitness[best_bee_index]

        # Abandon solution if no improvement after a certain number of trials
        trial_indices = np.where(employed_fitness >= best_fitness)
        abandoned_indices = np.random.choice(trial_indices[0], size=num_trials, replace=False)
        employed_bees[abandoned_indices] = np.random.uniform(bounds[0], bounds[1], size=(num_trials, num_dimensions))

    return best_solution, best_fitness

# Main function to run ABC optimization
def main():
    num_bees = 50
    num_iterations = 100
    num_trials = 5
    bounds = (-10, 10)

    best_solution, best_fitness = abc_optimization(num_bees, num_iterations, num_trials, bounds)

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

if __name__ == "__main__":
    main()
