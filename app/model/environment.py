# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:07:18 2023

@author: Florian Martin

"""
from model.grid import Grid
from math import exp, log, ceil, floor
from model.cell import HealthyCell, CancerCell, OARCell, Cell
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd


class GridEnv:
    
    def __init__(self, reward, h_threshold=4700, c_threshold=8900, sources = 100,
                 average_healthy_glucose_absorption = .36,
                 average_cancer_glucose_absorption = .54,
                 average_healthy_oxygen_consumption = 21.6,
                 average_cancer_oxygen_consumption = 21.6,
                 cell_cycle = [11, 8, 4, 1]):
        
        self.reward = reward
        self.time = 0
        self.xsize = 50
        self.ysize = 50
        self.hcells = 1000
        self.prob = self.hcells / (self.xsize * self.ysize)
        self.grid = None
        self.sources = sources
        
        self.oxygen_factor = 1 #average_cancer_oxygen_consumption/21.6
        self.glucose_factor = 1 #average_cancer_glucose_absorption/0.54
        
        
        self.nb_stages_cancer = 50
        self.nb_stages_healthy = 5
        self.nb_actions        = 4
        
        self.h_threshold = h_threshold
        self.c_threshold = c_threshold
        
        self.state_helper_hcells = self.h_threshold / (self.nb_stages_healthy - 2.0)
        self.state_helper_ccells = self.c_threshold / (self.nb_stages_cancer - 2.0)
         
        
        self.average_healthy_glucose_absorption = average_healthy_glucose_absorption
        self.average_cancer_glucose_absorption  = average_cancer_glucose_absorption
        
        self.average_healthy_oxygen_consumption = average_healthy_oxygen_consumption
        self.average_cancer_oxygen_consumption  = average_cancer_oxygen_consumption
        
        self.quiescent_glucose_level = average_healthy_glucose_absorption*2*24 #17.28 
        self.quiescent_oxygen_level = average_healthy_oxygen_consumption*2*24 #960
        
        self.critical_glucose_level = average_healthy_glucose_absorption*(3/4)*24 #6.48
        self.critical_oxygen_level = average_healthy_oxygen_consumption*(3/4)*24 #360
        
        self.cell_cycle = cell_cycle
         
    def reset(self):
    
        # Results
        self.nb = 2000
        self.time_arr    = np.arange(0, self.nb, 1)
        self.healthy_arr = np.array([np.nan]*self.nb)
        self.cancer_arr  = np.array([np.nan]*self.nb)
        self.dose_arr    = np.array([np.nan]*self.nb)
        
        self.total_dose = 0.
        
        self.glucose_arr = list()
        self.oxygen_arr = list()
        self.grid_arr = list()
        self.density_arr = list() 
        
        del self.grid
        HealthyCell.cell_count = 0
        CancerCell.cell_count = 0
        self.time = 0
        self.grid = Grid(self.xsize, self.ysize, self.sources,
                                             average_healthy_glucose_absorption=self.average_healthy_glucose_absorption,
                                             average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                                             average_healthy_oxygen_consumption=self.average_healthy_oxygen_consumption,
                                             average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption,
                                             critical_glucose_level=self.critical_glucose_level,
                                             critical_oxygen_level=self.critical_oxygen_level,
                                             quiescent_oxygen_level=self.quiescent_oxygen_level,
                                             quiescent_glucose_level=self.quiescent_glucose_level,
                                             cell_cycle=self.cell_cycle,
                                        oar=None)
        # Init Healthy Cells
        for i in range(self.xsize):
            for j in range(self.ysize):
                if random.random() < self.prob:
                    new_cell = HealthyCell(stage=random.randint(0, 4),
                                             average_healthy_glucose_absorption=self.average_healthy_glucose_absorption,
                                             average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                                             average_healthy_oxygen_consumption=self.average_healthy_oxygen_consumption,
                                             average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption,
                                             critical_glucose_level=self.critical_glucose_level,
                                             critical_oxygen_level=self.critical_oxygen_level,
                                             quiescent_oxygen_level=self.quiescent_oxygen_level,
                                             quiescent_glucose_level=self.quiescent_glucose_level)
                    self.grid.cells[i, j].append(new_cell)
        
        # Init Cancer Cell
        
        new_cell = CancerCell(stage=random.randint(0, 3),
                 average_healthy_glucose_absorption=self.average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption=self.average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption,
                 critical_glucose_level=self.critical_glucose_level,
                 critical_oxygen_level=self.critical_oxygen_level,
                 quiescent_oxygen_level=self.quiescent_oxygen_level,
                 quiescent_glucose_level=self.quiescent_glucose_level,
                 cell_cycle=self.cell_cycle)
        
        self.grid.cells[self.xsize//2, self.ysize//2].append(new_cell)

        self.grid.count_neighbors()
        
        # First : tumor growth and cells spreading
        self.init_hcell_count = HealthyCell.cell_count
        
    def go(self, steps=1):
        for _ in range(steps):
            # Storing current simulation state
            self.dose_arr[self.time] = self.total_dose
            self.healthy_arr[self.time] = HealthyCell.cell_count
            self.cancer_arr[self.time]  = CancerCell.cell_count
            self.glucose_arr.append(self.grid.glucose)
            self.oxygen_arr.append(self.grid.oxygen)
            self.grid_arr.append([[patch_type_color(self.grid.cells[i][j]) for j in range(self.grid.ysize)] for i in range(self.grid.xsize)])
            self.density_arr.append([[len(self.grid.cells[i][j]) for j in range(self.grid.ysize)] for i in range(self.grid.xsize)])
            
            
            
            self.grid.fill_source(130 * self.glucose_factor, 4500 * self.oxygen_factor)
            self.grid.cycle_cells()
            self.grid.diffuse_glucose(0.2)
            self.grid.diffuse_oxygen(0.2)
            self.time += 1
            if self.time % 24 == 0:
                self.grid.compute_center()
                
    def adjust_reward(self, dose, ccell_killed, hcells_lost):
        
        if self.inTerminalState():
            if self.end_type == "L" or self.end_type == "T":
                return -1
            else:
                if self.reward == 'dose':
                    return - dose / 400 + 0.5 - (self.init_hcell_count - HealthyCell.cell_count) / 3000
                else:
                    return 0.5 - (self.init_hcell_count - HealthyCell.cell_count) / 3000#(cppCellModel.HCellCount() / self.init_hcell_count) - 0.5 - (2 * hcells_lost/2500)
        else:
            if self.reward == 'dose' or self.reward == 'oar':
                return - dose / 400 + (ccell_killed - 5 * hcells_lost)/100000
            elif self.reward == 'killed':
                return (ccell_killed - 5 * hcells_lost)/100000
        
    def act(self, action):
        
        dose = action + 1
        print("dose of :", dose)
        self.total_dose += dose
        pre_hcell = HealthyCell.cell_count
        pre_ccell = CancerCell.cell_count # Previous State
        self.grid.irradiate(dose)
        m_hcell = HealthyCell.cell_count
        m_ccell = CancerCell.cell_count
        post_hcell = HealthyCell.cell_count # Next State
        post_ccell = CancerCell.cell_count
        
        return self.adjust_reward(dose, pre_ccell - post_ccell, pre_hcell-min(post_hcell, m_hcell))
        
    def inTerminalState(self):
        
        if CancerCell.cell_count <= 0 :
            self.end_type = 'W'
            return True
        elif HealthyCell.cell_count < 10:
            self.end_type = "L"
            return True
        elif self.time > 1550:
            self.end_type = "T"
            return True
        else:
            return False
        
    def observe(self):
        return HealthyCell.cell_count, CancerCell.cell_count

    def ccell_state(self, count):
        if count <= 10:
            return count
        if count < 500:
            div = (500-10)/15
            return int(ceil(count/div))+9
        if count > 7000:
            if count < 8000:
                return self.nb_stages_cancer - 2
            else: 
                return self.nb_stages_cancer - 1
        else: 
            div = (7000-500)/22
            return int(ceil(count/div))+24

    def hcell_state(self, count):
        #return min(self.nb_stages_healthy - 1, int(ceil(max(HealthyCell.cell_count - 9, 0) / self.state_helper_hcells)))
        return min(self.nb_stages_healthy - 1, max(0, int(ceil((count-(2875+375))/375))))

    def convert(self, obs):
        discrete_state = (self.ccell_state(obs[1]), self.hcell_state(obs[0]))
        return discrete_state
    
    
    def env_parameters(self):
        
        grid = Grid(self.xsize, self.ysize, self.sources,
                                             average_healthy_glucose_absorption=self.average_healthy_glucose_absorption,
                                             average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                                             average_healthy_oxygen_consumption=self.average_healthy_oxygen_consumption,
                                             average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption,
                                             critical_glucose_level=self.critical_glucose_level,
                                             critical_oxygen_level=self.critical_oxygen_level,
                                             quiescent_oxygen_level=self.quiescent_oxygen_level,
                                             quiescent_glucose_level=self.quiescent_glucose_level,
                                             cell_cycle=self.cell_cycle,
                                        oar=None)
        
        
        def latex_float(f):
            float_str = "{0:.4g}".format(f)
            if "e" in float_str:
                base, exponent = float_str.split("e")
                return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
            else:
                return float_str
        
        self.params = {}
        
        self.params["Starting healthy cells"] = [str(1000), "-", str(1000), str(self.hcells)]
        self.params["Starting cancer cells"] = [str(1), "-", str(1), str(1)]
        self.params["Starting nutrient sources"] = [str(100), "-", str(100), str(self.sources)]
        self.params["Starting glucose level"] = [latex_float(1e-6), "[mg]", str(100), str(grid.starting_glucose)]
        self.params["Starting oxygen level"] = [latex_float(1e-6), "[ml]", str(1000), str(grid.starting_oxygen)]
        
        
        self.params["Average glucose absorption (healthy)"] = [latex_float(3.6e-8), "mg/cell/hour", str(0.36), str(self.average_healthy_glucose_absorption)]
        self.params["Average glucose absorption (cancer)"] = [latex_float(5.4e-8), "mg/cell/hour", str(0.36), str(self.average_cancer_glucose_absorption)]
        self.params["Average oxygen consumption (healthy)"] = [latex_float(2.16e-8), "ml/cell/hour", str(20), str(self.average_healthy_oxygen_consumption)]
        self.params["Average oxygen consumption (cancer)"] = [latex_float(2.16e-8), "ml/cell/hour", str(20), str(self.average_cancer_oxygen_consumption)]
        
        self.params["Critical oxygen level"] = [latex_float(3.88e-8), "ml/cell", str(360), str(self.critical_oxygen_level)]
        self.params["Critical glucose level"] = [latex_float(6.48e-8), "mg/cell", str(6.48), str(self.critical_glucose_level)]
        self.params["Quiescent oxygen level"] = [latex_float(10.37e-8), "ml/cell", str(960), str(self.quiescent_oxygen_level)]
        self.params["Quiescent glucose level"] = [latex_float(1.728e-8), "mg/cell", str(17.28), str(self.quiescent_glucose_level)]

        
        self.df = pd.DataFrame.from_dict(data=self.params, orient="index", columns=["Theoretical Values", "Units", "Initial Model Values", "Modified Model Values"])
        
        print(self.df)
        print(self.df.to_latex(escape=False))
        
    
def patch_type_color(patch):
    if len(patch) == 0:
        return 0, 0, 0
    else:
        return patch[0].cell_color()
    
"""
env = GridEnv("dose", 5000, 13000,
              average_cancer_glucose_absorption=0.36)  
steps = 100  
env.reset()
while steps > 0:
    print(steps)
    
    while not env.inTerminalState() and steps > 0:
        state = env.convert(env.observe())
        action = 1
        reward = env.act(action)
        next_state = env.convert(env.observe())
        print(state, action, reward, next_state)
        
        #self.update(state, next_state, action, reward)
        steps -= 1
    if steps > 0:
        plt.imshow([[patch_type_color(env.grid.cells[i][j]) for j in range(env.grid.ysize)] for i in range(env.grid.xsize)])
        plt.show()
        env.reset()
        plt.imshow([[patch_type_color(env.grid.cells[i][j]) for j in range(env.grid.ysize)] for i in range(env.grid.xsize)])
        plt.show()
"""
