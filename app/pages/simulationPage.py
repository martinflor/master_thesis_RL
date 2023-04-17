import customtkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation 
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from scipy.stats import norm
import statistics
import threading
import time
import pickle
from model.environment import GridEnv
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
large = 22; med = 16; small = 12

params = {'axes.titlesize': large,
          'legend.fontsize': small,
          'figure.figsize': (5, 7),
          'axes.labelsize': small,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

dir_path = os.path.dirname(os.path.realpath(__file__))
        

class SimulationPage(customtkinter.CTkFrame):
    def __init__(self, master=None, params=None):
        self.master = master
        self.stop_event = threading.Event()
        self.is_paused = False  
        self.q_table = None
        
        for i in self.master.winfo_children():
            i.destroy()
            
        self.params = params
        self.name = self.params[4]
        self.path = self.params[5]
            
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Get the height of the taskbar
        taskbar_height = screen_height - self.master.winfo_rooty()
        
        print(screen_height - taskbar_height)
        
        # Set the window size and position
        self.master.geometry("%dx%d+0+0" % (screen_width, 760))
        
        self.structures = []
        """
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure((1, 2), weight=0)
        self.master.grid_rowconfigure((0, 2, 2), weight=1)
        """
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self.master, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Reinforcement Learning \n and \n Radiotherapy", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        
        # ZOOM ON PLOT BUTTON
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Zoom", command=self.focus_plot, width=190)
        self.sidebar_button_1.grid(row=1, column=0, padx=10, pady=10)
        self.focus = False
        
        # SAVE PLOT BUTTON
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Save Plot", width=190, command=self.save_plot)
        self.sidebar_button_2.grid(row=2, column=0, padx=10, pady=10)
        self.save = False
        
        # SAVE ANIMATION BUTTON
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Save Animation", width=190, command=self.save_anim)
        self.sidebar_button_3.grid(row=3, column=0, padx=10, pady=10)
        self.save = False
        
        # PAUSE SIMULATION BUTTON
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Pause", width=190, command=self.pause_simulation)
        self.sidebar_button_4.grid(row=4, column=0, padx=10, pady=10)
        
        # QUIT BUTTON
        
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Quit", fg_color="transparent", width=190, border_width=2, text_color=("gray10", "#DCE4EE"), command=self.quit_page)
        self.sidebar_button_5.grid(row=5, column=0, padx=10, pady=10)
        
        self.textbox = customtkinter.CTkTextbox(self.sidebar_frame)
        self.textbox.grid(row=6, column=0, rowspan=2, padx=20, pady=10, sticky="nsew")
        self.textbox.insert("0.0", "Help Box\n\n ZOOM BUTTON \n \n Zoom or Unzoom on the three bottom plots \n \n \n SAVE PLOT BUTTON \n \n Saving the entire graphic. The graphic will be found in the folder SAVE. \n \n \n SAVE ANIMATION BUTTON \n \n Saving the entire animation from the beggining to the current time step. It is deprecated to use this button without using the PAUSE button before. Be careful, it might take some minutes before finishing the save, do not close the window during this time. The animation will be found in the folder SAVE. \n \n \n PAUSE BUTTON \n \n Stop the simulation until the user press the button again.")
        self.textbox.configure(state="disabled", wrap="word")
        
        # LIGHT/DARK MODE
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], width=190,
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))   
        
        self.simulate()
        
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        
    def simulate(self):
        self.environment = GridEnv(reward="dose", sources = 100,
                 average_healthy_glucose_absorption = self.params[0],
                 average_cancer_glucose_absorption = self.params[1],
                 average_healthy_oxygen_consumption = self.params[2],
                 average_cancer_oxygen_consumption = self.params[3],
                 cell_cycle = self.params[6])
    
        self.environment.reset()
        self.environment.go(steps=1)
        self.idx = 0
        self.speed = 250
        
        self.structures.append(self.environment)
    
    
        x_pos = 0.2
        self.fig, axs = plt.subplots(2, 3, figsize = (16,12))
        canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().place(relx = x_pos, rely = 0.025, relwidth=0.75, relheight=0.775)
    
        self.cell_plot = axs[0][0]
        self.cell_density_plot = axs[0][1]
        self.glucose_plot = axs[0][2]
        #self.cellular_model = axs[0][3]
    
        self.dose_plot   = axs[1][0]
        self.healthy_plot = axs[1][1]
        self.cancer_plot  = axs[1][2]
    
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # CELL DENSITY COLORBAR
        self.div = make_axes_locatable(self.cell_density_plot)
        self.cax = self.div.append_axes('right', '5%', '5%')
        data = np.zeros((self.environment.xsize, self.environment.ysize))
        im = self.cell_density_plot.imshow(data)
        self.cb = self.fig.colorbar(im, cax=self.cax)
        
        # GLUCOSE COLORBAR
        self.div2 = make_axes_locatable(self.glucose_plot)
        self.cax2 = self.div2.append_axes('right', '5%', '5%')
        data = np.zeros((self.environment.xsize, self.environment.ysize))
        im = self.glucose_plot.imshow(data)
        self.cb2 = self.fig.colorbar(im, cax=self.cax2)
    
        self.nb = self.environment.nb
        self.time_arr    = self.environment.time_arr
        self.healthy_arr = self.environment.healthy_arr
        self.cancer_arr  = self.environment.cancer_arr
        self.dose_arr    = self.environment.dose_arr
        self.total_dose = self.environment.total_dose
        self.glucose_arr = self.environment.glucose_arr
        self.oxygen_arr = self.environment.oxygen_arr
        self.grid_arr = self.environment.grid_arr
        self.density_arr = self.environment.density_arr
    
        # Simulation Buttons
        
        
        self.slider = customtkinter.CTkSlider(self.master, from_=0, to=1550, width=1100)
        self.slider.bind("<ButtonRelease-1>", self.move_slider)
        self.slider.place(relx=x_pos, rely=0.875, anchor='nw')
        self.label_slider = customtkinter.CTkLabel(self.master, text='Simulation Time:')
        self.label_slider.place(relx=x_pos-0.025, rely=0.85, anchor='w')
        
        self.slider_speed = customtkinter.CTkSlider(self.master, from_=1, to=5, width=1100, command=self.move_slider_speed)
        self.slider_speed.place(relx=x_pos, rely=0.95, anchor='nw')
        self.label_slider_speed = customtkinter.CTkLabel(self.master, text='Simulation Speed:')
        self.label_slider_speed.place(relx=x_pos-0.025, rely=0.925, anchor='w')
        
        
        self.slider_speed.set(3)
        
        # call the update function after a delay
        self.update()
        
        
    def focus_plot(self):
        self.focus = not self.focus
        self.update_plot(self.idx)
        if self.sidebar_button_1.cget('text') =="Zoom":
            self.sidebar_button_1.configure(text="Unzoom")
        else:
            self.sidebar_button_1.configure(text="Zoom")
    
    def save_plot(self):
        self.save = True
        self.update_plot(self.idx)
        self.save = False
        
    def save_anim(self):
        self.anim = animation.FuncAnimation(self.fig, self.update_plot, 
							frames=len(self.structures), interval=100, repeat = False)
        
        self.anim.save('save/animated_env.gif', writer='imagemagick') 
        
    def save_env(self):
        with open('save/environment.pickle', 'wb') as file_env:
            pickle.dump(self.structures, file_env)
            
    def load_env(self):
        with open('save/environment.pickle', 'rb') as file:
                tmp_dict = pickle.load(file)
      
    def pause_simulation(self):
        self.is_paused = not self.is_paused
        if self.sidebar_button_4.cget('text') == "Pause":
            self.sidebar_button_4.configure(text="Continue")
        else:
            self.sidebar_button_4.configure(text="Pause")
        
    def stop_simulation(self):
        self.stop_event.set()
        self.is_running = False
    
    def quit_page(self):
        for i in self.master.winfo_children():
            i.destroy()
        self.master.quit()
        self.master.destroy()
        
    def load_q_table(self):
        self.q_table = np.load(self.path +  f'\\q_table_{int_from_str(self.path)}.npy', allow_pickle=False)
    
    def choose_action(self, state):
        if self.name == "Baseline":
            return 1.0
        else:
            if self.q_table is None:
                self.load_q_table()
            
            actions = np.argwhere(self.q_table[state]==np.max(self.q_table[state])).flatten()
            return np.random.choice(actions)
    
    def update(self):
        if not self.stop_event.is_set():
            if not self.is_paused:
                if not self.environment.inTerminalState():
                    self.environment.go(steps=1)
                    self.slider.set(self.idx)
                    if (self.idx > 349) and ((self.idx-326)%24 == 0):
                        state = self.environment.convert(self.environment.observe())
                        action = self.choose_action(state)
                        reward = self.environment.act(action)
                    self.update_plot(self.idx)

            self.master.after(self.speed, self.update)
        
    def update_plot(self, idx):
        
        self.time_arr    = self.environment.time_arr
        self.healthy_arr = self.environment.healthy_arr
        self.cancer_arr  = self.environment.cancer_arr
        self.dose_arr    = self.environment.dose_arr
        self.total_dose = self.environment.total_dose
        self.glucose_arr = self.environment.glucose_arr
        self.oxygen_arr = self.environment.oxygen_arr
        self.grid_arr = self.environment.grid_arr
        self.density_arr = self.environment.density_arr
        
        try:
            self.structures[idx] = self.environment
        except:
            self.structures.append(self.environment)

        
        self.plot_data(idx)
        
    def move_slider(self, event):
        tmp = self.speed
        self.speed = 0
        self.idx = int(self.slider.get())

        if self.idx > len(self.structures):
            self.idx = len(self.structures)
        self.slider.set(self.idx)
        self.environment = self.structures[self.idx-1]
        self.speed = tmp
        
    def move_slider_speed(self, value):
        self.slider_speed.set(int(value))
        lst = [1000, 500, 250, 100, 2]
        self.speed = lst[int(value)-1]
        
    def plot_data(self, i):
        
        def axes_off(ax):
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
          ax.spines['bottom'].set_visible(False)
          ax.spines['left'].set_visible(False)
          ax.tick_params(axis='both', which='both', length=0)
          
          return ax
    
        self.fig.suptitle('Cell proliferation at t = ' + str(i))    
    
        # plot cells
        self.cell_plot.clear()
        self.cell_plot.set_title("Cells")
        self.cell_plot.imshow(self.grid_arr[i], cmap='coolwarm')
        self.cell_plot.set_xticks([])
        self.cell_plot.set_yticks([])
        
        # plot cell density
        self.cax.cla()
        self.cell_density_plot.clear()
        self.cell_density_plot.set_title("Cell Density")
        im = self.cell_density_plot.imshow(self.density_arr[i], cmap='coolwarm')
        self.fig.colorbar(im, cax=self.cax)
        self.cell_density_plot.set_xticks([])
        self.cell_density_plot.set_yticks([])
        
        # plot glucose
        self.cax2.cla()
        self.glucose_plot.clear()
        self.glucose_plot.set_title("Glucose Concentration")
        im2 = self.glucose_plot.imshow(self.glucose_arr[i], cmap='YlOrRd')
        self.fig.colorbar(im2, cax=self.cax2)
        self.glucose_plot.set_xticks([])
        self.glucose_plot.set_yticks([])
        
        # plot dose
        self.dose_plot.clear()
        self.dose_plot = axes_off(self.dose_plot)
        self.dose_plot.set_title("Radiation Dose")
        self.dose_plot.plot(self.time_arr[:i+1], self.dose_arr[:i+1])
        
        # plot healthy cells
        self.healthy_plot.clear()
        self.healthy_plot = axes_off(self.healthy_plot)
        self.healthy_plot.set_title("Healthy Cells")
        self.healthy_plot.plot(self.time_arr[:i+1], self.healthy_arr[:i+1], label="Healthy", color="b")
        
        # plot cancer cells
        self.cancer_plot.clear()
        self.cancer_plot = axes_off(self.cancer_plot)
        self.cancer_plot.set_title("Cancer Cells")
        self.cancer_plot.plot(self.time_arr[:i+1], self.cancer_arr[:i+1], label="Cancer", color="r")
        
        if not self.focus:
            self.cancer_plot.set_xlim(0, self.time_arr[-1])
            self.healthy_plot.set_xlim(0, self.time_arr[-1])
            self.dose_plot.set_xlim(0, self.time_arr[-1])
        
        if self.save:
            plt.savefig("save/simulation.svg")
    
        self.idx += 1
        self.fig.canvas.draw()
    
def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())


if __name__ == '__main__':
    root=tk.Tk()
    #sv_ttk.set_theme("dark")
    app=Application(master=root)
    app.mainloop()



