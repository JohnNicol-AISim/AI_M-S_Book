import numpy as np

# Define the distance matrix for the TSP problem
distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Ant Colony Optimization (ACO) algorithm
def aco_tsp(num_ants, num_iterations, alpha, beta, rho):
    num_cities = len(distance_matrix)
    pheromone_matrix = np.ones((num_cities, num_cities))  # Initial pheromone matrix

    for _ in range(num_iterations):
        ant_routes = np.zeros((num_ants, num_cities), dtype=int)

        for ant in range(num_ants):
            visited = [0]  # Start from city 0
            unvisited = list(range(1, num_cities))

            for i in range(1, num_cities):
                probabilities = [
                    (pheromone_matrix[current_city][next_city] ** alpha) * (1.0 / distance_matrix[current_city][next_city] ** beta)
                    for next_city in unvisited
                ]
                probabilities = probabilities / np.sum(probabilities)
                next_city = np.random.choice(unvisited, p=probabilities)
                ant_routes[ant][i] = next_city
                visited.append(next_city)
                unvisited.remove(next_city)

            ant_routes[ant][-1] = 0  # Return to the starting city

        # Update pheromone matrix
        pheromone_matrix *= (1 - rho)

        for ant_route in ant_routes:
            for i in range(num_cities - 1):
                current_city, next_city = ant_route[i], ant_route[i + 1]
                pheromone_matrix[current_city][next_city] += 1 / distance_matrix[current_city][next_city]

    return ant_routes

# Main function to run ACO for TSP
def main():
    num_ants = 10
    num_iterations = 100
    alpha = 1.0
    beta = 5.0
    rho = 0.1

    best_routes = aco_tsp(num_ants, num_iterations, alpha, beta, rho)
    best_route_idx = np.argmin([sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) for route in best_routes])
    best_route = best_routes[best_route_idx]

    print("Best Route:", best_route)
    print("Total Distance:", sum(distance_matrix[best_route[i]][best_route[i + 1]] for i in range(len(best_route) - 1)))

if __name__ == "__main__":
    main()
