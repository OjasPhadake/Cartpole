import gymnasium as gym
import time

env = gym.make('CartPole-v1', render_mode='human')
obs, info = env.reset()

print("State vector: [cart_pos, cart_vel, pole_angle, pole_ang_vel]")
print(f"Initial: {obs}")

# --- Experiment 1: Push only LEFT (action=0) for 50 steps ---
print("\n--- Pushing LEFT only ---")
for step in range(50):
    action = 0  # push left
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")
    time.sleep(0.1)  # slow down for visualization
    
    if terminated or truncated:
        print(f"Episode finished after {step+1} steps")
        break
        
obs, info = env.reset()

print("\n--- Pushing RIGHT only ---")
for step in range(50):
    action = 1  # push right
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")
    time.sleep(0.1)  # slow down for visualization
    
    if terminated or truncated:
        print(f"Episode finished after {step+1} steps")
        break