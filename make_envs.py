import gym
import d4rl
import numpy as np
import os

def make_env(args, monitor=True):
    env = gym.make(args.env.name)
    return env
