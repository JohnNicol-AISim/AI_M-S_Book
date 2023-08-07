import numpy as np
import matplotlib.pyplot as plt

class Car:
    def __init__(self, position, speed):
        self.position = position
        self.speed = speed

    def update(self):
        self.position += self.speed

def main():
    num_cars = 10
    road_length = 100
    simulation_steps = 50

    # Initialize cars with random positions and speeds
    cars = [Car(np.random.randint(0, road_length), np.random.randint(1, 5)) for _ in range(num_cars)]

    positions_over_time = [[] for _ in range(num_cars)]
    for step in range(simulation_steps):
        for i, car in enumerate(cars):
            positions_over_time[i].append(car.position)
            car.update()

    # Plot the simulation results
    plt.figure(figsize=(10, 6))
    for i, positions in enumerate(positions_over_time):
        plt.plot(range(simulation_steps), positions, label=f"Car {i+1}")

    plt.xlabel("Simulation Steps")
    plt.ylabel("Car Position")
    plt.title("Agent-Based Simulation: Traffic Scenario")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
