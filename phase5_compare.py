import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def run_policy(policy_fn, n_episodes=20):
    """Run a hand-coded policy, return average reward."""
    env = gym.make("CartPole-v1", max_episode_steps=1000)
    total = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total += ep_reward
    env.close()
    return total / n_episodes

# Random policy
random_avg = run_policy(lambda obs: gym.make("CartPole-v1").action_space.sample())

# Hand-designed policy
def hand_policy(obs):
    _, _, theta, theta_dot = obs
    return 1 if (theta + 0.1 * theta_dot) > 0 else 0

hand_avg = run_policy(hand_policy)

# PPO
ppo_env = gym.make("CartPole-v1", max_episode_steps=1000)
model = PPO.load("cartpole_ppo", env=ppo_env)
ppo_mean, ppo_std = evaluate_policy(model, ppo_env, n_eval_episodes=20)
ppo_env.close()

# Print table
print("\n" + "="*40)
print(f"{'Controller':<20} {'Avg Reward':>10}")
print("="*40)
print(f"{'Random':<20} {random_avg:>10.1f}")
print(f"{'Hand-designed':<20} {hand_avg:>10.1f}")
print(f"{'PPO':<20} {ppo_mean:>10.1f}  (+/- {ppo_std:.1f})")
print("="*40)