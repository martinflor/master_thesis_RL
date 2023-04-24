# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:34:39 2023

@author: Robustness : cell cycle
"""

import os
import pandas as pd

list_dir = [file for file in os.listdir() if file.endswith('.txt')]

headers = ['Cell Cycle Duration [h]', 'Gap 1 [h]', 'Synthesis [h]', 'Gap 2 [h]', 'Mitosis [h]', 'TCP [%]', 
           'Fractions [-]', 'Duration [h]', 'Doses [Gy]', 'Survival [-]']
rows = []
fractions = []
duration = []
survival = []
dose = []

for idx, dirs in enumerate(list_dir):
    with open(list_dir[idx], 'r') as file:
        cc_file = file.readlines()
        
    if list_dir[idx] == 'summary.txt':
        continue
    
    G1 = int(cc_file[-12][8:10].strip())
    S = int(cc_file[-11][12:14].strip())
    G2 = int(cc_file[-10][8:10].strip())
    M = int(cc_file[-9][10])
    TCP = float(cc_file[-5][5:8])
    mean_frac, std_frac = float(cc_file[-4][26:31]), float(cc_file[-4][41:46])
    mean_duration, std_duration = float(cc_file[-3][18:24]), float(cc_file[-3][34:40])
    mean_survival, std_survival = float(cc_file[-2][18:24]), float(cc_file[-2][49:57])
    mean_dose, std_dose = float(cc_file[-1][24:29]), float(cc_file[-1][39:44])
    
    fractions.append(mean_frac)
    duration.append(mean_duration)
    survival.append(mean_survival)
    dose.append(mean_dose)
    rows.append([G1+S+G2+M ,G1, S, G2, M, TCP, f'{mean_frac} ' + '\u00B1' + f' {std_frac}',
                    f'{mean_duration} ' + '\u00B1' + f' {std_duration}',
                    f'{mean_dose} ' + '\u00B1' + f' {std_dose}',
                    f'{mean_survival} '])

df = pd.DataFrame(data=rows, columns=headers)

with open('summary.txt', 'w') as file:
    df_ = df.sort_values(by=['TCP [%]', 'Cell Cycle Duration [h]'])
    df_ = df_.reset_index(drop=True)
    file.write(df_.to_string())
    print(df_.to_latex(escape=False, index=False))
    
    
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotx
import matplotlib

def axes_off(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.tick_params(axis='both', which='both', length=0)
  
  return ax

s = [100_000/f for f in duration]

fig, ax = plt.subplots(figsize = (16,12))
ax = axes_off(ax)
ax.scatter(df["Cell Cycle Duration [h]"], df["TCP [%]"], s=s, label = "Q-learning")
ax.set_title("TCP w.r.t. the Cell Cycle Duration", fontsize=25)
ax.set_xlabel("Cell Cycle Duration (hours)", fontsize=15)
ax.set_ylabel("TCP (%)", fontsize=15)
ax.grid(axis='y', alpha=0.5)
plt.savefig("cell_cycle.svg", bbox_inches='tight')
plt.show()