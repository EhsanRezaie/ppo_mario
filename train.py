# ppo_mario_tensorboard.py
import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack ,VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# -------------------------
# مسیرها
# -------------------------
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# Callback برای checkpoint و لاگ TensorBoard
# -------------------------
class TensorboardCheckpointCallback(BaseCallback):
    """
    Callback برای ذخیره checkpoint دوره‌ای
    و ثبت لاگ‌ها برای TensorBoard
    """
    def __init__(self, checkpoint_freq: int = 20000, verbose=1):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ppo_mario_{self.n_calls}_steps")
            self.model.save(path)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model to {path}")
        # فقط ثبت لاگ TensorBoard، هیچ فریمی پردازش یا نمایش داده نمی‌شود
        return True

# -------------------------
# ساخت محیط Right-Only
# -------------------------
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)  
    return env

env = make_env()

# -------------------------
# بارگذاری مدل از checkpoint (در صورت نیاز)
# -------------------------
checkpoint_to_load = None  # مسیر checkpoint اگر میخوای ازش ادامه بدی
if checkpoint_to_load and os.path.exists(checkpoint_to_load + ".zip"):
    print(f"Loading model from {checkpoint_to_load}")
    model = PPO.load(checkpoint_to_load, env=env, tensorboard_log="./tensorboard/")
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
        device='cuda',
        tensorboard_log="./tensorboard/"
    )

# -------------------------
# تنظیمات آموزش
# -------------------------
total_timesteps = 100_000  # تعداد کل قدم‌ها
checkpoint_freq = 20_000   # هر چند قدم checkpoint ذخیره شود
callback = TensorboardCheckpointCallback(checkpoint_freq=checkpoint_freq)

timesteps_done = 0
batch_timesteps = 10_000  # حتماً بزرگتر یا مساوی checkpoint_freq باشه

# آموزش با tqdm
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
