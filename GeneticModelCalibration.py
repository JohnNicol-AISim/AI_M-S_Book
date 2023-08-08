import numpy as np

# Define the true model
def true_model(x, theta):
    return theta * x

# Generate synthetic data with noise
np.random.seed(42)
x_observed = np.linspace(0, 10, 100)
true_theta = 2.5
y_observed = true_model(x_observed, true_theta) + np.random.normal(0, 0.5, len(x_observed))

# Define the objective function for model calibration
def objective_function(theta, x_observed, y_observed):
    y_model = true_model(x_observed, theta)
    return np.sum((y_observed - y_model)**2)

# Genetic Algorithm for model calibration
def genetic_algorithm_calibration(x_observed, y_observed, num_generations=100, population_size=100, mutation_rate=0.1):
    population = np.random.uniform(low=0, high=5, size=population_size)
    for generation in range(num_generations):
        fitness_scores = [1 / (1 + objective_function(theta, x_observed, y_observed)) for theta in population]
        parents = np.random.choice(population, size=population_size//2, p=fitness_scores/np.sum(fitness_scores), replace=False)
        children = []
        for parent in parents:
            child = parent + np.random.uniform(-mutation_rate, mutation_rate)
            children.append(child)
        population = np.concatenate((parents, children))
    best_theta = population[np.argmax([objective_function(theta, x_observed, y_observed) for theta in population])]
    return best_theta

# Main function
def main():
    calibrated_theta = genetic_algorithm_calibration(x_observed, y_observed)

    print("True Theta:", true_theta)
    print("Calibrated Theta:", calibrated_theta)

if __name__ == "__main__":
    main()
