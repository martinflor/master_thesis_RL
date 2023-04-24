# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:34:39 2023

@author: Robustness : cell cycle
"""

import os
import pandas as pd

list_dir = [file for file in os.listdir() if file.endswith('.txt')]

headers = ['Sum of radiosensitivities', 'Gap 1', 'Synthesis', 'Gap 2', 'Mitosis', 'Quiescence', 'TCP [%]', 
           'Fractions [-]', 'Duration [h]', 'Doses [Gy]', 'Survival [-]', 'Start RT [h]', 'HCells [-]', 'CCells [-]']
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
    
    G1 = float(cc_file[-16][8:12].split()[0].strip())
    S = float(cc_file[-15][12:16].split()[0].strip())
    G2 = float(cc_file[-14][8:12].split()[0].strip())
    M = float(cc_file[-13][10:14].split()[0].strip())
    G0 = float(cc_file[-12][13:17].split()[0].strip())
    TCP = float(cc_file[-8][5:8].split()[0].strip())
    mean_frac, std_frac = float(cc_file[-7][26:31].split()[0].strip()), float(cc_file[-7][41:46].split()[0].strip())
    mean_duration, std_duration = float(cc_file[-6][18:24].split()[0].strip()), float(cc_file[-6][34:40].split()[0].strip())
    mean_survival, std_survival = float(cc_file[-5][18:24].split()[0].strip()), float(cc_file[-5][49:57].split()[0].strip())
    mean_dose, std_dose = float(cc_file[-4][24:29].split()[0].strip()), float(cc_file[-4][39:44].split()[0].strip())
    start_hour = int(cc_file[-3][24:27].strip())
    hcells = int(cc_file[-2][23:27].strip())
    ccells = int(cc_file[-1][21:27].strip())
    
    
    fractions.append(mean_frac)
    duration.append(mean_duration)
    survival.append(mean_survival)
    dose.append(mean_dose)
    rows.append([G1*11/24+S*8/24+G2*4/24+M/24 ,G1, S, G2, M, G0, TCP, f'{mean_frac} ' + '\u00B1' + f' {std_frac}',
                    f'{mean_duration} ' + '\u00B1' + f' {std_duration}',
                    f'{mean_dose} ' + '\u00B1' + f' {std_dose}',
                    f'{mean_survival} ',
                    start_hour,
                    hcells,
                    ccells])

df = pd.DataFrame(data=rows, columns=headers)

with open('summary.txt', 'w') as file:
    df_ = df.sort_values(by='Sum of radiosensitivities')
    df_ = df_.reset_index(drop=True)
    file.write(df_.to_string())
    #print(df_.to_latex(escape=False, index=False))
    
    
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


fig, ax = plt.subplots(figsize = (16,12))
ax = axes_off(ax)
ax.scatter(df['Sum of radiosensitivities'], df["TCP [%]"], label = "Q-learning")
ax.set_title("TCP w.r.t. the total radiosensitivity", fontsize=25)
ax.set_xlabel('Sum of radiosensitivities', fontsize=15)
ax.set_ylabel("TCP (%)", fontsize=15)
ax.grid(axis='y', alpha=0.5)
plt.savefig("radiosensitivities.svg", bbox_inches='tight')
plt.show()