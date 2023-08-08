import random

# Define the Sphere function for optimization
def sphere_function(x):
    return sum(xi**2 for xi in x)

# Particle class to represent an individual particle in the swarm
class Particle:
    def __init__(self, dim):
        self.position = [random.uniform(-5.12, 5.12) for _ in range(dim)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dim)]
        self.best_position = self.position
        self.best_score = sphere_function(self.position)

# Particle Swarm Optimization algorithm
def pso_optimization(max_iterations, num_particles, dim):
    # Initialize the swarm with random particles
    swarm = [Particle(dim) for _ in range(num_particles)]

    # Find the global best position and score
    global_best_position = min(swarm, key=lambda particle: particle.best_score).best_position
    global_best_score = sphere_function(global_best_position)

    # Main optimization loop
    for _ in range(max_iterations):
        for particle in swarm:
            # Update particle's velocity and position
            for i in range(dim):
                r1, r2 = random.random(), random.random()
                particle.velocity[i] = 0.5 * particle.velocity[i] + 2 * r1 * (particle.best_position[i] - particle.position[i]) + 2 * r2 * (global_best_position[i] - particle.position[i])
                particle.position[i] += particle.velocity[i]

            # Update particle's best position and score
            new_score = sphere_function(particle.position)
            if new_score < particle.best_score:
                particle.best_position = particle.position
                particle.best_score = new_score

            # Update global best position and score
            if new_score < global_best_score:
                global_best_position = particle.position
                global_best_score = new_score

    return global_best_position, global_best_score

# Main function to run the PSO optimization
def main():
    max_iterations = 100
    num_particles = 50
    dim = 2

    best_position, best_score = pso_optimization(max_iterations, num_particles, dim)

    print("Optimal solution found:")
    print("Best Position:", best_position)
    print("Best Score:", best_score)

if __name__ == "__main__":
    main()
