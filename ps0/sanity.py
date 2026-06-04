from env import ApartmentEnv
from policies import RandomPolicy

env = ApartmentEnv(4, 4)
policy = RandomPolicy(4)

obs, _ = env.reset()

terminated = False
while not terminated:
    week = obs["week"]
    quality = obs["quality"]
    action = policy.act(obs)

    obs, reward, terminated, _, _ = env.step(action)

    print(f"({week}, {quality}, {action}, {reward}, {terminated})")

env.close()
