import PyCANDY
import time

import torch

from intelli_timestamp_generator import *
import numpy as np
import collections
from torch_helper import *
import os
import argparse
from SAFERL.CPQ import CPQ
from SAFERL.utils import ReplayBuffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--constraint_threshold", default=0.9, type=float)
    parser.add_argument("--alpha", default=10)
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    args = parser.parse_args()
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    state_dim =26
    action_dim = 9
    max_action = 8
    policy = CPQ(state_dim, action_dim, max_action, discount=args.discount,threshold=args.constraint_threshold, alpha=args.alpha)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_csv()
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
    #save_dir = f"./results/{args.discount}_{args.constraint_threshold}.txt"
    #eval_log = open(save_dir, 'w')
    #start training
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
        #average_return, discounted_cost, _ = eval_policy(policy, args.env, args.seed, mean, std, args.constraint_threshold, discount=args.discount)
        #eval_log.write(f'{t + 1},{average_return},{discounted_cost}\n')
        #eval_log.flush()

        # self.actor = Actor(state_dim, action_dim, max_action).to(device)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        #
        # self.reward_critic = Double_Critic(state_dim, action_dim).to(device)
        # self.reward_critic_target = copy.deepcopy(self.reward_critic)
        # self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=3e-4)
        #
        # self.cost_critic = Critic(state_dim, action_dim).to(device)
        # self.cost_critic_target = copy.deepcopy(self.cost_critic)
        # self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=3e-4)
        #
        # self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        # self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

    torch.save(policy.actor, "./models/actor.pt")
    torch.save(policy.cost_critic, "./models/cost_critic.pt")
    torch.save(policy.cost_critic_target, "./models/cost_critic_target.pt")
    torch.save(policy.reward_critic, "./models/reward_critic.pt")
    torch.save(policy.reward_critic_target, "./models/reward_critic_target.pt")
    torch.save(policy.vae, "./models/vae.pt")
