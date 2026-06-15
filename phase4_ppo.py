import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# --- Train ---
env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",   # Policy is a small neural network
    env,
    verbose=1      # Print training progress
)

print("Training PPO for 50,000 timesteps...")
model.learn(total_timesteps=50_000)

model.save("cartpole_ppo")
print("Model saved.")

env.close()

# --- Evaluate ---
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"\nPPO Mean Reward: {mean_reward:.1f} +/- {std_reward:.1f}")
eval_env.close()

# --- Watch it ---
watch_env = gym.make("CartPole-v1", render_mode="human")
obs, info = watch_env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = watch_env.step(action)
    done = terminated or truncated

watch_env.close()