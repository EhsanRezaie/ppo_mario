# dqn_mario_bestmodel.py
import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# -------------------------
# مسیرها
# -------------------------
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TENSORBOARD_LOG = "./tensorboard/"

# -------------------------
# Callback برای checkpoint دوره‌ای
# -------------------------
class TensorboardCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_freq: int = 20000, verbose=1):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0:
            path = os.path.join(CHECKPOINT_DIR, f"dqn_mario_{self.n_calls}_steps")
            self.model.save(path)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model to {path}")
        return True

# -------------------------
# Callback برای ذخیره بهترین مدل
# -------------------------
class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -float('inf')
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # میانگین reward آخرین episode‌ها
            mean_reward = self.training_env.get_attr('episode_rewards')[0]
            if mean_reward and len(mean_reward) > 0:
                mean_reward_val = sum(mean_reward)/len(mean_reward)
                if mean_reward_val > self.best_mean_reward:
                    self.best_mean_reward = mean_reward_val
                    path = os.path.join(self.log_dir, 'best_model')
                    self.model.save(path)
                    if self.verbose > 0:
                        print(f"[BestModel] New best mean reward {mean_reward_val:.2f}, model saved to {path}")
        return True

# -------------------------
# ساخت محیط Right-Only با frame stack و monitor
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
checkpoint_to_load = None
if checkpoint_to_load and os.path.exists(checkpoint_to_load + ".zip"):
    print(f"Loading model from {checkpoint_to_load}")
    model = DQN.load(checkpoint_to_load, env=env, tensorboard_log=TENSORBOARD_LOG)
else:
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=10_000,
        learning_starts=500,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        device='cuda',
        tensorboard_log=TENSORBOARD_LOG
    )

# -------------------------
# تنظیمات آموزش
# -------------------------
total_timesteps = 100_000
checkpoint_freq = 20_000

checkpoint_callback = TensorboardCheckpointCallback(checkpoint_freq=checkpoint_freq)
best_model_callback = SaveBestModelCallback(check_freq=checkpoint_freq, log_dir=CHECKPOINT_DIR)

timesteps_done = 0
batch_timesteps = 10_000

# آموزش با tqdm
with tqdm(total=total_timesteps, desc="Training Mario DQN") as pbar:
    while timesteps_done < total_timesteps:
        model.learn(
            total_timesteps=batch_timesteps,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, best_model_callback]
        )
        timesteps_done += batch_timesteps
        pbar.update(batch_timesteps)

# -------------------------
# ذخیره نهایی مدل
# -------------------------
final_model_path = os.path.join(CHECKPOINT_DIR, "dqn_mario_final")
model.save(final_model_path)
print(f"Training finished. Final model saved to {final_model_path}")
