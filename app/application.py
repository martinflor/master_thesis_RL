import customtkinter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from scipy.stats import norm
import statistics
import time
import pickle
import webbrowser
from pages.simulationPage import SimulationPage
from model.environment import GridEnv
from pages.helpPage import help_page
import os
import pandas as pd

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

dir_path = os.path.dirname(os.path.realpath(__file__))

def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("RL and RT")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        taskbar_height = screen_height - self.winfo_rooty()
        
        self.geometry("%dx%d+0+0" % (screen_width, 760))

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Reinforcement Learning \n and \n Radiotherapy", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Start", command=self.simulation, width=190)
        self.sidebar_button_1.grid(row=1, column=0, padx=10, pady=10)
        
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Quit", width=190, fg_color="transparent", text_color=("gray10", "#DCE4EE"), border_width=2, command=self.quit)
        self.sidebar_button_4.grid(row=2, column=0, padx=10, pady=10)
        
        # create textbox
        self.textbox = customtkinter.CTkTextbox(self.sidebar_frame)
        #self.textbox.grid(row=3, column=0, rowspan=4, padx=20, pady=10, sticky="nsew")
        self.textbox.place(relx=0.05, rely=0.25, relwidth=0.9, relheigh=0.45)
        
        with open(dir_path + '\\misc\\treatment_help.txt', 'r') as file:
            treatment_file = file.readlines()
            
        with open(dir_path + '\\misc\\cell_cycle_help.txt', 'r') as file:
            cell_cycle_file = file.readlines()
            
        with open(dir_path + '\\misc\\nutrients_help.txt', 'r') as file:
            nutrients_file = file.readlines()
            
        treatment_file = ''.join(line for line in treatment_file)
        cell_cycle_file = ''.join(line for line in cell_cycle_file)
        nutrients_file = ''.join(line for line in nutrients_file)
                                                            
        
        self.texts = {"Nutrients" : "Help Box\n\n\n " + nutrients_file,
                      "Treatment" : "Help Box\n\n\n " + treatment_file,
                      "Cell cycle" : "Help Box\n\n\n " + cell_cycle_file,
                      "Radiosensitivity" : "Help Box\n\n\n"}
        self.textbox.insert('0.0', self.texts["Treatment"])
        self.textbox.configure(state="disabled", wrap="word")


        # EPL LOGO
        
        epl = customtkinter.CTkImage(light_image=Image.open("images/EPL.jpg"),
                                  dark_image=Image.open("images/EPL.jpg"),
                                  size=(150, 80))
        button_epl = customtkinter.CTkButton(self, text= '', 
                                                image=epl, fg_color='transparent')
        button_epl.place(relx=1, rely=1, anchor='se')
        
        # GITHUB ICON
        
        github = customtkinter.CTkImage(light_image=Image.open("images/github.png"),
                                  dark_image=Image.open("images/github.png"),
                                  size=(30, 30))
        button_github = customtkinter.CTkButton(self.sidebar_frame, text= 'GITHUB', 
                                                image=github, fg_color='transparent', text_color=('black', 'white'),
                                                command=self.open_github)
        #button_github.grid(row=6, column=0, padx=20, pady=(10, 0))
        button_github.place(relx=0.025, rely=0.75, relwidth=0.9, relheight=0.05)
        
        # LINKEDIN ICON
        
        linkedin = customtkinter.CTkImage(light_image=Image.open("images/linkedin.png"),
                                  dark_image=Image.open("images/linkedin.png"),
                                  size=(30, 30))
        button_linkedin = customtkinter.CTkButton(self.sidebar_frame, text= 'LinkedIn', 
                                                image=linkedin, fg_color='transparent', text_color=('black', 'white'),
                                                command=self.open_linkedin)
        button_linkedin.place(relx=0.025, rely=0.825, relwidth=0.9, relheight=0.05)
        
        # AUTHORS
        
        self.author_label = customtkinter.CTkLabel(self, text="Author: Florian Martin")
        self.author_label.place(relx=0.16, rely=.975, anchor='sw')
        
        self.supervisor_label = customtkinter.CTkLabel(self, text='Supervisors: Mélanie Ghislain, Manon Dausort, Damien Dasnoy-Sumell, Benoît Macq')
        self.supervisor_label.place(relx=0.16, rely=1.0, anchor='sw')
        
        # APPEARANCE MODE
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], width=190,
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=550, command=self.update_helpbox)
        self.tabview.place(relx= 0.17, rely=0.025, relwidth=0.82, relheight=0.8)
        
        # PROGRESS BAR
        
        self.progressbar_1 = customtkinter.CTkProgressBar(self)
        self.progressbar_1.place(relx= 0.17, rely=0.83, relwidth=0.82)
        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.tabview.add("Treatment")
        self.tabview.add("Nutrients")
        self.tabview.add("Cell cycle")
        self.tabview.add("Radiosensitivity")
        self.tabview.tab("Treatment").grid_columnconfigure(0, weight=1)  
        self.tabview.tab("Nutrients").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Cell cycle").grid_columnconfigure(0, weight=1)
        

        
        # NUTRIENTS
        self.fields = ('Average healthy glucose absorption', 'Average cancer glucose absorption',
                  'Average healthy oxygen consumption', 'Average cancer oxygen consumption')
        
        self.default = [.36, .54, 20, 20]
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), frameon=False)
        self.fig.patch.set_alpha(0)
        self.plot_data(self.default)
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Nutrients"))
        canvas.draw()
        canvas.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas.get_tk_widget().place(relx=0.45, rely=0.01, relwidth=0.55, relheight=0.99)
        
        self.labels = [customtkinter.CTkLabel(self.tabview.tab("Nutrients"), width=60, text=field+": ", anchor='nw', font=customtkinter.CTkFont(size=16, weight="bold")) for field in self.fields]
        self.entries = [customtkinter.CTkEntry(self.tabview.tab("Nutrients")) for _ in range(4)]
        
        init_x, init_y = 0.05, 0.1
        change_x, change_y = 0.25, 0.1
        
        for idx, ent in enumerate(self.entries):
            self.entries[idx].insert(1, str(self.default[idx]))
            self.entries[idx].bind('<KeyRelease>', self.update_plot)
            self.labels[idx].place(relx=init_x, rely=init_y+change_y*idx)
            self.entries[idx].place(relx=init_x+change_x, rely=init_y+change_y*idx)

        self.table = self.make_table(self.default)
        self.table.place(relx = 0.1, rely = 0.55)

        # TREATMENT
        
        self.tabview_tt = customtkinter.CTkTabview(self.tabview.tab("Treatment"), width=550)
        self.tabview_tt.place(relx=0.45, rely=0.01, relwidth=0.55, relheight=.99)
        
        self.tabview_tt.add("Performances")
        self.tabview_tt.add("Agent's q-table")

        self.tabview_tt.tab("Performances").grid_columnconfigure(0, weight=1)  
        self.tabview_tt.tab("Agent's q-table").grid_columnconfigure(0, weight=1)
        
        values = self.list_agent()
        lst = [i for i, _ in values]
        
        self.combobox_label = customtkinter.CTkLabel(self.tabview.tab("Treatment"), text="RL Agent:", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.combobox_label.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)
        self.combobox_1 = customtkinter.CTkComboBox(self.tabview.tab("Treatment"),
                                                    values=lst, command=self.update_description)
        self.combobox_1.place(relx=0.025, rely=0.1, relwidth=0.25, relheight=0.05)
        
        self.states_label = customtkinter.CTkLabel(self.tabview.tab("Treatment"), text="Number of unexplored states by the agent : /", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.states_label.place(relx=0.015, rely=0.85, relwidth=0.4, relheight=0.05)
        
        
        # Treatment : Performances
        
        self.fig_box, self.axes = plt.subplots(3,1, figsize=(24,20))
        self.fig_box.patch.set_alpha(0)
        canvas_box = FigureCanvasTkAgg(self.fig_box, master=self.tabview_tt.tab("Performances"))
        canvas_box.draw()
        canvas_box.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas_box.get_tk_widget().place(relx=0.01, rely=0.01, relwidth=0.99, relheight=0.99)
        
        # Treatment : Q-table
        
        self.fig_table, self.axes_table = plt.subplots(4, 1, constrained_layout=True, figsize = (16,12))
        self.fig_table.patch.set_alpha(0)
        canvas_table = FigureCanvasTkAgg(self.fig_table, master=self.tabview_tt.tab("Agent's q-table"))
        canvas_table.draw()
        canvas_table.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas_table.get_tk_widget().place(relx=0.01, rely=0.01, relwidth=0.99, relheight=0.99)

        self.update_description(1)
        
        # Cell Cycle
        self.data = [11, 8, 4, 1, 24]
        self.fig2, self.ax = plt.subplots(2,1, figsize=(6, 9), frameon=False)
        self.fig2.patch.set_alpha(0)
        
        canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tabview.tab("Cell cycle"))
        canvas2.draw()
        canvas2.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas2.get_tk_widget().place(relx=0.45, rely=0.01, relwidth=0.55, relheight=0.99)
        
        
        self.fields_cc = ('Gap 1 (G1)', 'Synthesis (S)', 'Gap 2 (G2)', 'Mitosis (M)', 'Cell Cycle Duration')
        self.labels_cc = [customtkinter.CTkLabel(self.tabview.tab("Cell cycle"), width=60, text=field+": ", anchor='nw', font=customtkinter.CTkFont(size=16, weight="bold")) for field in self.fields_cc]
        self.entries_cc = [customtkinter.CTkEntry(self.tabview.tab("Cell cycle")) for _ in range(len(self.fields_cc))]
        
        init_x, init_y = 0.05, 0.1
        change_x, change_y = 0.25, 0.075
        
        for idx, ent in enumerate(self.entries_cc):
            self.entries_cc[idx].insert(1, str(self.data[idx]))
            self.entries_cc[idx].bind('<KeyRelease>', self.update_plot2)
            self.labels_cc[idx].place(relx=init_x, rely=init_y+change_y*idx)
            self.entries_cc[idx].place(relx=init_x+change_x, rely=init_y+change_y*idx)
        
        self.entry_text = tk.StringVar()
        self.entries_cc[idx].configure(textvariable=self.entry_text)
        self.plot_pie(self.data)
        
        
        # Radiosensitivity
        
        self.radio = [1, .75, 1.25, 1.25, .75, 0,96875]
        self.fig_radio, self.ax_radio = plt.subplots(1, 1, figsize=(12,9))
        self.fig_radio.patch.set_alpha(0)
        
        canvas_radio = FigureCanvasTkAgg(self.fig_radio, master=self.tabview.tab("Radiosensitivity"))
        canvas_radio.draw()
        canvas_radio.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas_radio.get_tk_widget().place(relx=0.45, rely=0.01, relwidth=0.55, relheight=0.99)
        
        self.fields_radio = ('Gap 1 (G1)', 'Synthesis (S)', 'Gap 2 (G2)', 'Mitosis (M)', 'Quiescent (G0)', 'Sum of radiosensitivities')
        self.labels_radio = [customtkinter.CTkLabel(self.tabview.tab("Radiosensitivity"), width=60, text=field+": ", anchor='nw', font=customtkinter.CTkFont(size=16, weight="bold")) for field in self.fields_radio]
        self.entries_radio = [customtkinter.CTkEntry(self.tabview.tab("Radiosensitivity")) for _ in range(len(self.fields_radio))]
        
        init_x, init_y = 0.05, 0.1
        change_x, change_y = 0.25, 0.075
        
        for idx, ent in enumerate(self.entries_radio):
            self.entries_radio[idx].insert(1, str(self.radio[idx]))
            self.entries_radio[idx].bind('<KeyRelease>', self.update_plot_radio)
            self.labels_radio[idx].place(relx=init_x, rely=init_y+change_y*idx)
            self.entries_radio[idx].place(relx=init_x+change_x, rely=init_y+change_y*idx)
        
        self.entry_text2 = tk.StringVar()
        self.entries_radio[idx].configure(textvariable=self.entry_text2)
        self.plot_radio(self.radio)
        
    def get_values_radio(self):
        values = []
        for idx, field in enumerate(self.fields_radio):
            values.append(float(self.entries_radio[idx].get()))
        return values
        
    def update_plot_radio(self, event):
        self.after(500, self.plot_radio, self.get_values_radio())
        
    def plot_radio(self, radiosensitivities):
        self.ax_radio.clear()
        alpha_norm_tissue = 0.15
        beta_norm_tissue = 0.03
        dose = np.linspace(0,10, 1000)
        f = lambda x, stage : np.exp(radiosensitivities[stage] * (-alpha_norm_tissue*x - beta_norm_tissue * (x ** 2)))
        
        for idx, rad in enumerate(radiosensitivities[:-1]):
            survival = f(dose, idx)
            self.ax_radio.plot(dose, survival, label=f"radiosensitivity of {rad} ({self.fields_radio[idx]})")
            self.ax_radio.set_xlabel("Radiation dose (Gy)")
            self.ax_radio.set_ylabel("Surviving fraction")
        
        self.ax_radio.legend()
        
        sum_radio = (11*radiosensitivities[0]+8*radiosensitivities[1]+4*radiosensitivities[2]+radiosensitivities[3])/24
        
        self.entries_radio[-1].configure(state='normal')
        self.entry_text2.set(str(sum_radio))
        self.entries_radio[-1].configure(state='disabled')
        
        self.fig_radio.canvas.draw()

    def update_helpbox(self):
        self.textbox.configure(state="normal", wrap="word")
        tab = self.tabview.get()
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", self.texts[tab])
        self.textbox.configure(state="disabled", wrap="word")
        
        
    def get_values2(self):
        values = []
        for idx, field in enumerate(self.fields_cc):
            values.append(int(self.entries_cc[idx].get()))
        return values
        
    def update_plot2(self, event):
        self.after(500, self.plot_pie, self.get_values2())

    def plot_pie(self, data):
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct
        
        default = [11, 8, 4, 1, 24]
        labels = ['Gap 1', 'Synthesis', 'Gap 2', 'Mitosis']
        colors = sns.color_palette('pastel')[0:4]
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].pie(default[:-1], labels = labels, colors = colors, autopct=make_autopct(default[:-1]))
        self.ax[1].pie(data[:-1], labels = labels, colors = colors, autopct=make_autopct(data[:-1]))
        plt.tight_layout()
        
        self.entries_cc[-1].configure(state='normal')
        self.entry_text.set(str(sum(data[:-1])))
        self.entries_cc[-1].configure(state='disabled')
        
        self.fig2.canvas.draw()

    
    def open_github(self):
        webbrowser.open_new_tab('https://github.com/martinflor/master_thesis_RL')

    def open_linkedin(self):
        webbrowser.open_new_tab('https://www.linkedin.com/in/florian-martin-554350239/')

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        
    def get_values(self):
        values = []
        for idx, field in enumerate(self.fields):
            values.append(float(self.entries[idx].get()))
        return values
        
    def update_plot(self, event):
        self.after(500, self.plot_data, self.get_values())
        self.update_table(self.get_values())
    
    def make_table(self, values):
        table_frame = customtkinter.CTkFrame(self.tabview.tab("Nutrients"), fg_color='transparent')
        table_frame.grid(row=len(self.fields)+1, column=1, padx=10, pady=10, sticky='n')
        
        
        self.padx_table = 25 
        
        headers = ['Parameter', 'Value']
        
        rows = [('Quiescent (Glucose)', f"{2*24*values[0]:.3f}"), 
                ('Quiescent (Oxygen)', f"{2*24*values[2]:.3f}"),
                ('Critical (Glucose)', f"{(3/4)*24*values[0]:.3f}"),
                ('Critical (Oxygen)', f"{(3/4)*24*values[2]:.3f}")]
        
        for j, header in enumerate(headers):
            label = customtkinter.CTkLabel(table_frame, text=header, font=customtkinter.CTkFont(size=18, weight="bold"))
            label.grid(row=0, column=j, padx=self.padx_table, pady=5)
        
        for row, data in enumerate(rows):
            for col, item in enumerate(data):
                label = customtkinter.CTkLabel(table_frame, text=item, font=('Arial', 14))
                label.grid(row=row+1, column=col, padx=self.padx_table, pady=5, sticky='w')
                
        return table_frame
    
    def update_table(self, values):
        # delete the children of the table_frame widget
        for widget in self.table.winfo_children():
            widget.destroy()
            
        
        headers = ['Parameter', 'Value']
        
        rows = [('Quiescent (Glucose)', f"{2*24*values[0]:.3f}"), 
                ('Quiescent (Oxygen)', f"{2*24*values[2]:.3f}"),
                ('Critical (Glucose)', f"{(3/4)*24*values[0]:.3f}"),
                ('Critical (Oxygen)', f"{(3/4)*24*values[2]:.3f}")]
        
        for j, header in enumerate(headers):
            label = customtkinter.CTkLabel(self.table, text=header, font=customtkinter.CTkFont(size=18, weight="bold"))
            label.grid(row=0, column=j, padx=self.padx_table, pady=5)
        
        for row, data in enumerate(rows):
            for col, item in enumerate(data):
                label = customtkinter.CTkLabel(self.table, text=item, font=('Arial', 14))
                label.grid(row=row+1, column=col, padx=self.padx_table, pady=5, sticky='w')
        
    def update_description(self, event):
        agent_name = self.combobox_1.get()  # get the selected agent name
        print(agent_name)
        self.description(self.tabview.tab("Treatment"), agent_name)
        

    def description(self, menu, file_name):
            
        self.agent_frame = customtkinter.CTkFrame(menu, fg_color='transparent')
        self.agent_frame.place(relx=0.01, rely=0.25, relwidth=0.4, relheight=0.5)
        
        tmp_dict = self.get_agent(dir_path + '\\TabularAgentResults\\results_baseline.pickle')
        tcp_baseline = tmp_dict["TCP"]
        fractions_baseline = (np.mean(tmp_dict["fractions"]), np.std(tmp_dict["fractions"]))
        doses_baseline = (np.mean(tmp_dict["doses"]), np.std(tmp_dict["doses"]))
        duration_baseline = (np.mean(tmp_dict["duration"]), np.std(tmp_dict["duration"]))
        survival_baseline = (np.mean(tmp_dict["survival"]), np.std(tmp_dict["survival"]))
        
        if file_name == 'Baseline':
            tcp = tcp_baseline
            fractions = fractions_baseline
            doses = doses_baseline
            duration = duration_baseline
            survival = survival_baseline
        else:
            file_list = self.list_agent()
            for name, path_ in file_list:
                if file_name == name:
                    path = path_
                
            tmp_dict = self.get_agent(path + f'\\results_{int_from_str(path)}.pickle')
            tcp = tmp_dict["TCP"]
            fractions = (np.mean(tmp_dict["fractions"]), np.std(tmp_dict["fractions"]))
            doses = (np.mean(tmp_dict["doses"]), np.std(tmp_dict["doses"]))
            duration = (np.mean(tmp_dict["duration"]), np.std(tmp_dict["duration"]))
            survival = (np.mean(tmp_dict["survival"]), np.std(tmp_dict["survival"]))
    
        # create table headers
        headers = ['', file_name, 'Baseline']
            
        # create table rows
        
        data = [('TCP', f"{tcp}", f"{tcp_baseline}"), 
                ('Fractions', f"{fractions[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{fractions[1]:.3f}",
                 f"{fractions_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{fractions_baseline[1]:.3f}"), 
                ('Doses', f"{doses[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{doses[1]:.3f}",
                 f"{doses_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{doses_baseline[1]:.3f}"), 
                ('Duration', f"{duration[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{duration[1]:.3f}",
                 f"{duration_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{duration_baseline[1]:.3f}"),
                ('Survival', f"{survival[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{survival[1]:.3f}",
                 f"{survival_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{survival_baseline[1]:.3f}")]

    
        padx_value = 30
        for j, header in enumerate(headers):
                    label = customtkinter.CTkLabel(self.agent_frame, text=header, font=('Arial', 18))
                    label.grid(row=0, column=j*4, padx=padx_value, pady=5)
        
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                label = customtkinter.CTkLabel(self.agent_frame, text=cell, font=('Arial', 14))
                label.grid(row=i+1, column=j*4, padx=padx_value, pady=5)
                
        self.boxplot_agent(tmp_dict["fractions"], tmp_dict["duration"], tmp_dict["survival"], file_name)
        if file_name != 'Baseline':
            self.q_table_agent(path)

                
    def get_agent(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def list_agent(self):
        
        lst = [('Baseline', '')]
        filename = dir_path + "\\TabularAgentResults\\"
        list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
        
        for name, path in list_dir:
            subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
            for i in range(len(subfolders)):
                lst.append((name + ' ' + int_from_str(subfolders[i]), subfolders[i]))
                
        return lst
    
    def q_table_agent(self, path):
        
        def get_q_color(value, vals):
            if all(x==max(vals) for x in vals):
                return "grey", 0.5
            if value == max(vals):
                return "green", 1.0
            else:
                return "red", 0.3

        filename = path + f'\\q_table_{int_from_str(path)}'
        q_table = np.load(filename + '.npy', allow_pickle=False)
        
        self.axes_table[0].clear()
        self.axes_table[1].clear()
        self.axes_table[2].clear()
        self.axes_table[3].clear()
        
        self.axes_table[0].set_title("Action 1 : 1 Gray")
        self.axes_table[1].set_title("Action 2 : 2 Grays")
        self.axes_table[2].set_title("Action 3 : 3 Grays")
        self.axes_table[3].set_title("Action 4 : 4 Grays")
        
        count = 0
        for x, x_vals in enumerate(q_table):
                for y, y_vals in enumerate(x_vals):
                    self.axes_table[0].scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                    self.axes_table[1].scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                    self.axes_table[2].scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])
                    self.axes_table[3].scatter(x, y, c=get_q_color(y_vals[3], y_vals)[0], marker="o", alpha=get_q_color(y_vals[3], y_vals)[1])
                
                    if all(x==y_vals[0] for x in y_vals):
                        count += 1
                        
        self.states_label.configure(text=f"Number of unexplored states : {count}")
        self.fig_table.canvas.draw()
    
    def boxplot_agent(self, fractions, duration, survival, name):
        
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()
        
        self.fig_box.suptitle(name)

        # Create a DataFrame for each list
        data_fractions = pd.DataFrame({"Values": fractions})
        data_fractions["Type"] = "Fractions [-]"
        
        data_duration = pd.DataFrame({"Values": duration})
        data_duration["Type"] = "Duration \n [hours]"
        
        data_survival = pd.DataFrame({"Values": survival})
        data_survival["Type"] = "Survival [-]"
        
        # Combine the three DataFrames
        data = pd.concat([data_fractions, data_duration, data_survival], ignore_index=True)
        
        # Create the subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        
        # Loop through each subplot and create the boxplot with scatter points
        for i, data_type in enumerate(["Fractions [-]", "Duration \n [hours]", "Survival [-]"]):
            sns.boxplot(x="Values", y="Type", orient='h', data=data[data["Type"] == data_type], ax=self.axes[i], palette="Set2", width=0.5)
            sns.stripplot(x="Values", y="Type", orient='h', data=data[data["Type"] == data_type], ax=self.axes[i], color=".25")
            self.axes[i].set_ylabel("")

        self.fig_box.canvas.draw()
    
    def quit_page(self):
        self.quit()
        self.destroy()
        
    def plot_data(self, values):
        self.ax1.clear()
        self.ax2.clear()
    
        f1 = lambda x, y: min(x, y)
        f2 = lambda x, y: max(x, y)
    
        # Plot between -10 and 10 with .001 steps.
        x_axis = np.arange(-0.5, 2.5, 0.0001)
    
        # Calculating mean and standard deviation
        mean = 1
        sd = 1 / 3
    
        normal = norm.pdf(x_axis, mean, sd)
        results = []
    
        for i in range(len(x_axis)):
            results.append(max(0, min(2, normal[i])))
    
        self.ax1.plot(x_axis, np.array(results) * values[1], label="Cancer Cells")
        self.ax1.plot(x_axis, np.array(results) * values[0], label="Healthy Cells")
        self.ax1.set_ylabel("Glucose Absorption")
        self.ax1.grid(alpha=0.5)
        self.ax1.legend()
    
        self.ax2.plot(x_axis, np.array(results) * values[3], label="Cancer Cells")
        self.ax2.plot(x_axis, np.array(results) * values[2], label="Healthy Cells")
        self.ax2.set_ylabel("Oxygen Consumption")
        self.ax2.grid(alpha=0.5)
        self.ax2.legend()
    
        # Set the background color of the plot to transparent
        self.ax1.set_facecolor('none')
        self.ax2.set_facecolor('none')
    
        # Remove the border of the plot
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['bottom'].set_visible(False)
        self.ax1.spines['left'].set_visible(False)
    
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['bottom'].set_visible(False)
        self.ax2.spines['left'].set_visible(False)
    
        self.fig.canvas.draw()

        
    def simulation(self):
        file_name = self.combobox_1.get()
        file_list = self.list_agent()
        for name, path_ in file_list:
            if file_name == name:
                path = path_
                
        params = self.get_values() + [file_name, path, self.get_values2(), self.get_values_radio()]
        self.simulation = SimulationPage(self, params)


if __name__ == "__main__":
    app = App()
    app.mainloop()