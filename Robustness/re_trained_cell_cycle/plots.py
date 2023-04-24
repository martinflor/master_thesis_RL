# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:25:44 2023

@author: Florian Martin

"""


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import pickle
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


class AgentResults:
    def __init__(self, agent, nb):
        
        print(agent)
        self.nb = nb
        self.results = []
        self.fractions = []
        self.doses = []
        self.duration = []
        self.TCPs = []
        self.survivals = []
        self.rewards = {}
        self.dicts = []
        self.epsilons = []
        self.alphas = []
    
        for i in range(nb):
            with open(dir_path + f"\\{agent}\\{i}\\results_{i}" + '.pickle', 'rb') as file:
                tmp_dict = pickle.load(file)
                self.dicts.append(tmp_dict)
                self.fractions.append(np.mean(tmp_dict["fractions"]))
                self.doses.append(np.mean(tmp_dict["doses"]))
                self.duration.append(np.mean([t for t in tmp_dict["duration"]]))
                self.TCPs.append(tmp_dict["TCP"])
                self.survivals.append(np.mean(tmp_dict["survival"]))
                self.results.append(tmp_dict)
                self.epsilons.append(tmp_dict["epsilon"])
                if "rewards" in tmp_dict.keys():
                    self.rewards[i] = tmp_dict["rewards"]
                    
                print(i, f'Mean : {np.mean(tmp_dict["survival"])}, Std : {np.std(tmp_dict["survival"])}')
            
     
    
        self.dict_ = {"fractions" : self.fractions, 
                      "doses" : self.doses, 
                      "duration" : self.duration, 
                      "TCP" : self.TCPs, 
                      "survival" : self.survivals,
                      "epsilon" : self.epsilons,
                      }
    
    def get_results(self):
        return self.dict_
    
    def print_results(self):
        df = pd.DataFrame.from_dict(self.dict_)
        print(df)
        return df
        
    def print_mean_std(self):
        df = pd.DataFrame.from_dict(self.dict_)
        print(df.mean())
        print("\n")
        print(df.std())
        
    def print_mean_std_TCP(self):
        df = pd.DataFrame.from_dict(self.dict_)
        df = df[df["TCP"] == 100.0]
        df.reset_index(inplace=True)
        print(df.mean())
        print("\n")
        print(df.std())
        
    def print_best_results(self):
        df = pd.DataFrame.from_dict(self.dict_)
        df = df[df["TCP"] == 100.0]
        df.reset_index(inplace=True)
    
        print(df)
        print(f"Smallest fractions : {np.min(df.fractions)} at index {np.argmin(df.fractions)}")
        print(f"Smallest dose : {np.min(df.doses)} at index {np.argmin(df.doses)}")
        print(f"Smallest duration : {np.min(df.duration)} at index {np.argmin(df.duration)}")
        return df

"""
Q_results = AgentResults(agent="QAgent", nb=20)
df_Q = Q_results.print_results()
df_Q["Agent"] = ["Q-Agent"]*len(df_Q)
"""

Sarsa_results = AgentResults(agent="SARSAgent", nb=16)
df_sarsa = Sarsa_results.print_results()
df_sarsa["Agent"] = ["Sarsa"]*len(df_sarsa)
"""
Exp_Sarsa_results = AgentResults(agent="ExpSARSAgent", nb=20)
df_exp_sarsa = Exp_Sarsa_results.print_results()
df_exp_sarsa["Agent"] = ["ExpSarsa"]*len(df_exp_sarsa)
"""
#df = pd.concat([df_Q, df_sarsa, df_exp_sarsa])
#df = df[df["TCP"] == 100.0]
df = df_sarsa.copy()
df.reset_index(inplace=True)
#df.drop(columns=["level_0"], inplace=True)

print(df)
print(f"Smallest fractions : {np.min(df.fractions)} at index {np.argmin(df.fractions)}")
print(f"Smallest dose : {np.min(df.doses)} at index {np.argmin(df.doses)}")
print(f"Smallest duration : {np.min(df.duration)} at index {np.argmin(df.duration)}")
print(f"Biggest Survival : {np.max(df.survival)} at index {np.argmax(df.survival)}")



import matplotx



plt.figure(figsize = (32,32))

df_TCP = df[df["TCP"] == 100.0]

sns.relplot(x=df["duration"], y=df["survival"]*100, size=df["TCP"], hue=df["Agent"], 
            height=8, aspect=1.25, sizes=(10, 150), palette="plasma")
plt.title("Survival Percentage w.r.t. duration treatment time", fontsize=25)
plt.xlabel("Treatment Duration [hours]", fontsize=15)
plt.ylabel("Survival Percentage [%]", fontsize=15)
plt.grid(axis='y', alpha=0.5)
plt.yticks(np.arange(90,100, 1))
plt.savefig("duration_survival.svg", bbox_inches='tight')
plt.show()



plt.figure(figsize = (16,12))
#plt.scatter(df_Q["duration"], df_Q["doses"], s=df_Q["TCP"], label = "Q-learning")
plt.scatter(df_sarsa["duration"], df_sarsa["doses"], s=df_sarsa["TCP"], label="Sarsa")
#plt.scatter(df_exp_sarsa["duration"], df_exp_sarsa["doses"], s=df_exp_sarsa["TCP"], label="ExpSarsa")
plt.title("Number of doses w.r.t. duration treatment time", fontsize=25)
plt.xlabel("Treatment Duration [hours]", fontsize=15)
plt.ylabel("Doses [Gy]", fontsize=15)
plt.grid(axis='y', alpha=0.5)
plt.legend(fontsize=18)
plt.savefig("duration_doses.svg", bbox_inches='tight')
plt.show()

