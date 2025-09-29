from hydra import initialize, compose
import numpy as np
import hydra


with initialize(config_path="./calvin_env/conf/"):
  cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
  cfg.env["use_egl"] = False
  cfg.env["show_gui"] = False
  cfg.env["use_vr"] = False
  cfg.env["use_scene_info"] = True
  print(cfg.env)
env = hydra.utils.instantiate(cfg.env)


# env = get_env("data/calvin_D/validation", show_gui=False)
observation = env.reset()

data = np.load("data/calvin_D_cat_30hz/episode_0000000.npz", allow_pickle=True)
actions = data["actions"]
rgb_static = data["rgb_static"]

observations = []
for a in actions:
    a = (a[:3], a[3:6], a[6:])
    observation, reward, done, info = env.step(a)
    for key, value in observation.items():
        print(f"{key}: {value.shape}")



