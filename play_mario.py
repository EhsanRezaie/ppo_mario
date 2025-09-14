# play_mario_dqn.py
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import numpy as np

MODEL_PATH = "./checkpoints/dqn_mario_final"  # مسیر مدل DQN

# -------------------------
# محیط واحد برای مدل و نمایش
# -------------------------
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=2)  # برای DQN معمولا n_stack=2 کافی است
    return env

env = make_env()
model = DQN.load(MODEL_PATH, env=env)

obs = env.reset()
done = False

while not done:
    # انتخاب اکشن مدل
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    frame = env.render(mode='rgb_array')  # تصویر واقعی بازی
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_bgr, (1280, 720))
    cv2.imshow("Super Mario DQN Play", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

env.close()
cv2.destroyAllWindows()
