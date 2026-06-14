import gymnasium as gym

# Making an environment open in human mode makes it possible to visualize in real time
env= gym.make('CartPole-v1', render_mode='human')

obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space     :", env.action_space)
print("Initial obs      :", obs)

for step in range(1000):
    
    action = env.action_space.sample()  # take a random action
    obs, reward, terminated, truncated, info = env.step(action) # take a step in the environment
    
    print(f"Step {step}: action={action}, obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}",)
    
    if terminated or truncated:
        print(f"Episode finished after {step+1} steps")
        obs, info = env.reset()  # reset the environment for the next episode

env.close()