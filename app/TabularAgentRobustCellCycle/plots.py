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
    def __init__(self, nb):
        
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
        self.counts = []
        
        self.list_dir = [(f.name, f.path) for f in os.scandir(dir_path) if f.is_dir() and nb in f.name]
    
        for i in range(len(self.list_dir)):
            with open(self.list_dir[i][1] + f'\\results_{self.list_dir[i][0]}' + '.pickle', 'rb') as file:
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
                    
                self.q_table = np.load(self.list_dir[i][1] + f'\\q_table_{self.list_dir[i][0]}' + '.npy', allow_pickle=False)
                count = 0
                for x, x_vals in enumerate(self.q_table):
                    for y, y_vals in enumerate(x_vals):
                        if all(x==y_vals[0] for x in y_vals):
                            count += 1
                            
                self.counts.append(count)
                    
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



Sarsa16 = AgentResults('16')
df16 = Sarsa16.print_results()
names = [x[0] for x in Sarsa16.list_dir]
df16["Cell Cycle"] = [16] * len(names)

Sarsa18 = AgentResults('18')
df18 = Sarsa18.print_results()
names = [x[0] for x in Sarsa18.list_dir]
df18["Cell Cycle"] = [18] * len(names)

Sarsa20 = AgentResults('20')
df20 = Sarsa20.print_results()
names = [x[0] for x in Sarsa20.list_dir]
df20["Cell Cycle"] = [20] * len(names)

df = pd.concat([df16, df18, df20])
df.reset_index(inplace=True)

print(df)
print(f"Smallest fractions : {np.min(df.fractions)} at index {np.argmin(df.fractions)}")
print(f"Smallest dose : {np.min(df.doses)} at index {np.argmin(df.doses)}")
print(f"Smallest duration : {np.min(df.duration)} at index {np.argmin(df.duration)}")
print(f"Biggest Survival : {np.max(df.survival)} at index {np.argmax(df.survival)}")



import matplotx



plt.figure(figsize = (32,32))

df_TCP = df[df["TCP"] == 100.0]

sns.relplot(x=df["duration"], y=df["survival"]*100, size=df["TCP"], hue=df["Cell Cycle"], 
            height=8, aspect=1.25, sizes=(10, 150), palette="plasma")
plt.title("Survival Percentage w.r.t. duration treatment time", fontsize=25)
plt.xlabel("Treatment Duration [hours]", fontsize=15)
plt.ylabel("Survival Percentage [%]", fontsize=15)
plt.grid(axis='y', alpha=0.5)
plt.savefig("duration_survival.svg", bbox_inches='tight')
plt.show()



plt.figure(figsize = (16,12))
sns.scatterplot(x=df["duration"], y=df["doses"], size=df["TCP"], hue=df["Cell Cycle"])
plt.title("Number of doses w.r.t. duration treatment time", fontsize=25)
plt.xlabel("Treatment Duration [hours]", fontsize=15)
plt.ylabel("Doses [Gy]", fontsize=15)
plt.grid(axis='y', alpha=0.5)
plt.legend(fontsize=18)
plt.savefig("duration_doses.svg", bbox_inches='tight')
plt.show()


y = Sarsa16.counts
x = np.arange(0, len(y), 1)

y2 = Sarsa18.counts
x2 = np.arange(0, len(y2), 1)

y3 = Sarsa20.counts
x3 = np.arange(0, len(y3), 1)

plt.figure(figsize = (16,12))
plt.plot(x,y, label = 'Cell Cycle : 16 hours')
plt.plot(x2,y2, label = 'Cell Cycle : 18 hours')
plt.plot(x3,y3, label = 'Cell Cycle : 20 hours')
plt.title("Number of unexplored states by the agent w.r.t. training epochs", fontsize=25)
plt.xlabel("Training epochs", fontsize=15)
plt.xticks(ticks=x)
plt.ylabel("Number of unexplored states", fontsize=15)
plt.grid(axis='y', alpha=0.5)
plt.legend()
plt.savefig("unexplored.svg")
plt.show()

