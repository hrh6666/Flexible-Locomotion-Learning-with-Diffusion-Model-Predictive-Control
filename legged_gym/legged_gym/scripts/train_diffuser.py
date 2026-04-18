import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from legged_gym.debugger import break_into_debugger

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)

if __name__ == '__main__':
    args = get_args()
    train(args)