"""
ppo_mario_training_logger.py
نسخه بهبود یافته با logger و نمودار reward دوره‌ای
ویژگی‌ها:
- Right-Only اکشن‌ها
- نوار پیشرفت با tqdm
- ذخیره دوره‌ای checkpoint
- ذخیره لاگ reward و نمودار دوره‌ای در logger/
- امکان لود از checkpoint
"""

import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# مسیرها
# -------------------------
CHECKPOINT_DIR = "./checkpoints"
LOGGER_DIR = "./logger"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGGER_DIR, exist_ok=True)

# -------------------------
# Callback برای checkpoint و logging
# -------------------------
class LoggerCallback(BaseCallback):
    """
    Callback برای ذخیره checkpoint و لاگ reward دوره‌ای
    """
    def __init__(self, check_freq: int = 10000, log_freq: int = 5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_freq = log_freq
        self.rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        # ذخیره checkpoint
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ppo_mario_{self.n_calls}_steps")
            self.model.save(path)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model to {path}")

        # ذخیره لاگ و نمودار reward دوره‌ای
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                self.rewards.append(mean_reward)
                self.steps.append(self.n_calls)

                # ذخیره لاگ متنی
                log_path = os.path.join(LOGGER_DIR, "reward_log.txt")
                with open(log_path, "w") as f:
                    for s, r in zip(self.steps, self.rewards):
                        f.write(f"{s},{r}\n")

                # ذخیره نمودار (فقط توی فایل، بدون نمایش)
                plt.figure(figsize=(10,5))
                plt.plot(self.steps, self.rewards, marker='o', linestyle='-')
                plt.xlabel("Step")
                plt.ylabel("Mean Reward")
                plt.title("Training Reward (Updated)")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(LOGGER_DIR, "reward_plot.png"))
                plt.close()

                if self.verbose > 0:
                    print(f"[Logger] Updated reward plot at step {self.n_calls}")
        return True
# -------------------------
# ساخت محیط Right-Only
# -------------------------
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env

env = make_env()

# -------------------------
# بارگذاری مدل از checkpoint (در صورت نیاز)
# -------------------------
checkpoint_to_load = None  # مسیر checkpoint اگر میخوای ازش ادامه بدی
if checkpoint_to_load and os.path.exists(checkpoint_to_load + ".zip"):
    print(f"Loading model from {checkpoint_to_load}")
    model = PPO.load(checkpoint_to_load, env=env)
else:
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        gamma=0.95,
        clip_range=0.2,
        device='cuda'
    )

# -------------------------
# آموزش با نوار پیشرفت و callback
# -------------------------
total_timesteps = 100_000  # تعداد کل قدم‌های آموزش
check_freq = 10_000        # ذخیره checkpoint هر چند قدم
log_freq = 5_000           # لاگ و نمودار reward هر چند قدم

callback = LoggerCallback(check_freq=check_freq, log_freq=log_freq)

timesteps_done = 0
batch_timesteps = 10_000  # حتماً بزرگتر یا مساوی check_freq باشه

with tqdm(total=total_timesteps, desc="Training Mario PPO") as pbar:
    while timesteps_done < total_timesteps:
        model.learn(
            total_timesteps=batch_timesteps,
            reset_num_timesteps=False,
            callback=callback
        )
        timesteps_done += batch_timesteps
        pbar.update(batch_timesteps)

# -------------------------
# ذخیره نهایی مدل
# -------------------------
final_model_path = os.path.join(CHECKPOINT_DIR, "ppo_mario_final")
model.save(final_model_path)
print(f"Training finished. Final model saved to {final_model_path}")

# -------------------------
# نمودار نهایی reward
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(callback.steps, callback.rewards)
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.title("Final Training Reward")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(LOGGER_DIR, "reward_plot_final.png"))
plt.close()
