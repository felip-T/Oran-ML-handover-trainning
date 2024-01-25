import ns3ai_gym_env
import gymnasium as gym
import traceback
import sys
import PPO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

target="model3"
ns3Path="/home/felipe/Downloads/ns-allinone-3.40/fork"

ns3Settings = {
    'use-train': 'true',
    'sim-time': 50,
    'seed': 1,
    # 'plot': 'true',
    # 'verbose': 'true',
    }

env = gym.make("ns3ai_gym_env/Ns3-v0", targetName=target, ns3Path=ns3Path, ns3Settings=ns3Settings)
obs_dim = env.observation_space
action_dim = env.action_space
print(f"obs_dim: {obs_dim.shape[0]}")
print(f"action_dim: {action_dim.shape}")

ppo = PPO.PPO(obs_dim.shape[0], 4, 0.0003, 0.001, 0.99, 5, 0.2, False, 0.6)

log_f = open("log.txt", "a")
log_f.write("Starting training\n")

# ppo.load("./saves")

try:
    for i in range(100):
        cur_reward = 0
        ns3Settings['seed'] = i+1
        obs, info = env.reset(options=ns3Settings)
        count = 0
        while True:
            action = ppo.select_action(obs)
            print("action: ", action)
            # print("stepping")
            obs, reward, done, _,info = env.step(action)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)
            print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")
            print(f'reward: {reward}')
            cur_reward += reward
            # print("acc reward: ", cur_reward)
            # print("Done stepping")
            if done:
                print("breaking")
                break
            count += 1
            if (count > 9):
                ppo.update()
                count = 0
        print("updating")
        ppo.update()
        ppo.save(f"./save/point{i}")
        log_f.write("Episode: {}, total reward: {}\n".format(i, cur_reward))
        log_f.flush()
        print("Episode: {}, total reward: {}".format(i, cur_reward))


except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Exception occurred: {}".format(e))
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)

log_f.close()

env.close()
