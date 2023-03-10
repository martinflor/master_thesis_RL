# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:39:05 2023

@author: Florian Martin

"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation 
import seaborn as sns
import imageio
import random
import numpy as np
import os
import pickle
from environment import GridEnv
from grid import Grid
from math import exp, log, ceil, floor
from cell import HealthyCell, CancerCell, OARCell


nb_stages_cancer = 50
nb_stages_healthy = 5
nb_actions        = 4

#dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dir_path = "/linux/martinflor/RL-Radiotherapy/TabularAgentResults/"
     
class Baseline:
    
    def __init__(self, env,
                 anim_simu=False, anim_q_table=False, anim_results=False):
        
        self.env = env
        self.nb_stages_healthy = nb_stages_healthy
        self.nb_stages_cancer  = nb_stages_cancer
        self.nb_actions        = nb_actions
        
        self.anim_simu = anim_simu
    
    def test(self, episodes):
        lengths_arr = []
        fracs_arr = []
        doses_arr = []
        survivals_arr = []
        
        doses_per_hour = {}
        fracs_per_hour = {}
        rewards = []
        
        sum_w = 0
        for ep in range(episodes):
            self.env.reset()
            sum_r = 0
            fracs = 0
            doses = 0
            time = 0
            doses_per_hour[ep] = {}
            init_hcell = HealthyCell.cell_count
            while not self.env.inTerminalState():
                state = self.env.convert(self.env.observe())
                action = 1
                reward = self.env.act(action)

                print(action + 1, "grays, reward =", reward)
                fracs += 1
                doses += action + 1
                doses_per_hour[ep][time] = action + 1
                time += 24
                sum_r += reward
                next_state = self.env.convert(self.env.observe())
                
                
            if self.env.end_type == 'W':
                sum_w += 1
            
            fracs_arr.append(fracs)
            doses_arr.append(doses)
            lengths_arr.append(time)
            survival = HealthyCell.cell_count / init_hcell
            survivals_arr.append(survival)
            rewards.append(sum_r)

            
        self.epochs_arr = np.arange(episodes)
        self.fracs_arr = np.array(fracs_arr)
        self.doses_arr = np.array(doses_arr)
        self.lengths_arr = np.array(lengths_arr)
        self.survivals_arr = np.array(survivals_arr)
        self.rewards = np.array(rewards)
        self.tcp = 100.0 *sum_w/episodes

        print("TCP: " , self.tcp)
        print("Average num of fractions: ", np.mean(self.fracs_arr), " std dev: ", np.std(self.fracs_arr))
        print("Average radiation dose: ", np.mean(self.doses_arr), " std dev: ", np.std(self.doses_arr))
        print("Average duration: ", np.mean(self.lengths_arr), " std dev: ", np.std(self.lengths_arr))
        print("Average survival: ", np.mean(self.survivals_arr), " std dev: ", np.std(self.survivals_arr))
        
        results = {"TCP" : self.tcp,
                    "fractions" : self.fracs_arr,
                   "doses" : self.doses_arr,
                   "duration" : self.lengths_arr,
                   "survival" : self.survivals_arr,
                   "doses_per_hour" : doses_per_hour,
                   "rewards" : self.rewards
                   }
        
        return results
    
    def run(self, test_steps):
        filename = dir_path + "/baseline/"

        self.results = self.test(episodes=test_steps)
            self.save(str(i), self.results)
            
        with open(filename+name+"/results_" + name + ".pickle", 'wb') as file:
            pickle.dump(results, file)
        
         
          
        