# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:36:26 2023

@author: Florian Martin

"""

import argparse
import datetime
print(datetime.datetime.now())

parser = argparse.ArgumentParser(description='2D Env. with Tabular Agent')

#parser.add_argument("mode", type = str, choices = ["train", "test"])
parser.add_argument("agent", type = str, help="Tabular Agent to train/test", choices = ["QAgent", "SARSAgent", "ExpSARSAgent"])
parser.add_argument("--reward", type = str, help="Reward", choices = ["dose", "killed"], default="dose")
parser.add_argument("--animation", type = bool, nargs=3, default=[False, False, False])
parser.add_argument("--epochs", type = int, help="Number of epochs, train steps, test steps", nargs = 3, default = [20, 2500, 100])
parser.add_argument("--gamma", type = float, help="Discount Factor", default = 0.95)
parser.add_argument("--alpha", type = float, help="Learning Rate", default = 0.8)
parser.add_argument("--epsilon", type = float, help="epsilon greedy policy argument", default = 0.8)
parser.add_argument("--final_alpha", type = float, help="Final value of learning rate", default = 0.5)
parser.add_argument("--final_epsilon", type = float, help="Final value of epsilon", default = 0.05)

# Cellular Model Parameters

parser.add_argument("--h_threshold", type = float, default = 13_000)
parser.add_argument("--c_threshold", type = float, default = 5_000)
parser.add_argument("--sources", type = float, default = 100)

parser.add_argument("--average_healthy_glucose_absorption", type = float, default = .36)
parser.add_argument("--average_cancer_glucose_absorption", type = float, default = .54)
parser.add_argument("--average_healthy_oxygen_consumption", type = float, default = 21.6)
parser.add_argument("--average_cancer_oxygen_consumption", type = float, default = 21.6)



args = parser.parse_args()
print(args)



from simulation import QAgent, SARSAgent, ExpSARSAgent
from environment import GridEnv

env = GridEnv(reward = args.reward, h_threshold = args.h_threshold, c_threshold = args.c_threshold, sources = args.sources,
                 average_healthy_glucose_absorption = args.average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption = args.average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption = args.average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption = args.average_cancer_oxygen_consumption)
    
if args.agent == "QAgent":
    
    agent = QAgent(env,
               args.gamma,
               args.alpha,
               args.epsilon,
               anim_simu=args.animation[0],
               anim_q_table=args.animation[1],
               anim_results=args.animation[2])
    
elif args.agent == "SARSAgent":
    
    agent = SARSAgent(env,
               args.gamma,
               args.alpha,
               args.epsilon,
               anim_simu=args.animation[0],
               anim_q_table=args.animation[1],
               anim_results=args.animation[2])

elif args.agent == "ExpSARSAgent":
    
    agent = ExpSARSAgent(env,
               args.gamma,
               args.alpha,
               args.epsilon,
               anim_simu=args.animation[0],
               anim_q_table=args.animation[1],
               anim_results=args.animation[2])


agent.run(args.epochs[0], args.epochs[1], args.epochs[2], args.final_epsilon, args.final_alpha)