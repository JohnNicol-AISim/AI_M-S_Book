import gym

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Q-learning algorithm
def q_learning(num_episodes, learning_rate, discount_factor, exploration_prob):
    q_table = {}
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Explore or exploit based on exploration probability
            if np.random.uniform(0, 1) < exploration_prob:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table.get(state, np.zeros(env.action_space.n)))
            
            # Take the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value using Q-learning update rule
            q_table[state] = q_table.get(state, np.zeros(env.action_space.n))
            q_table[state][action] = q_table[state][action] + learning_rate * (reward + discount_factor * np.max(q_table.get(next_state, np.zeros(env.action_space.n))) - q_table[state][action])
            
            state = next_state
    
    return q_table

# Train the Q-learning agent
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.1
q_table = q_learning(num_episodes, learning_rate, discount_factor, exploration_prob)

# Test the trained agent
state = env.reset()
done = False
total_reward = 0
while not done:
    action = np.argmax(q_table.get(state, np.zeros(env.action_space.n)))
    state, reward, done, _ = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)

env.close()
