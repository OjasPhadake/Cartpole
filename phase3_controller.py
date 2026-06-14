import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human") 
# The max num of timesteps is 500 by default, but we can set it to 1000 to give our policies more time to succeed
# This can be done by env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=1000)

def simple_policy(obs):
    """Look only at the angle"""
    
    cart_pos, cart_vel, theta, theta_dot = obs
    
    if theta > 0:
        return 1  # push right
    else:
        return 0  # push left
    
def better_policy(obs):
    """Look at the angle and angular velocity"""
    
    cart_pos, cart_vel, theta, theta_dot = obs
    
    if theta + 0.01 * theta_dot > 0:
        # Can try weights like 0.001, 0.5, 2, 100, etc to see how it affects the performance
        # The idea is to push in the direction of the angle, but also to counteract the angular velocity if it's large
        
        return 1  # push right
    else:
        return 0  # push left`
    
# --- Evaluate a policy over N episodes ---
def evaluate_policy(policy_fn, num_episodes=10, render=True):
    total_reward = 0.0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            done = terminated or truncated
        
        total_reward += episode_reward
        print(f"Episode {episode+1}: Reward={episode_reward}")
    
    avg_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

# print("=== Simple Policy (angle only) ===")
# avg_simple = evaluate_policy(simple_policy, num_episodes=10)

print("\n=== Better Policy (angle + rate) ===")
avg_better = evaluate_policy(better_policy, num_episodes=10)

env.close()