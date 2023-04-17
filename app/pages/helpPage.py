import customtkinter
import matplotlib.pyplot as plt
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
import os

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

dir_path = os.path.dirname(os.path.realpath(__file__))

def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())

class help_page(customtkinter.CTk):
    def __init__(self, master):
        # configure window
        self.master = master
        
        for i in self.master.winfo_children():
            i.destroy()
            
        self.master.title("RL and RT")
        
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Get the height of the taskbar
        taskbar_height = screen_height - self.master.winfo_rooty()
        
        print(screen_height - taskbar_height)
        
        # Set the window size and position
        self.master.geometry("%dx%d+0+0" % (screen_width, 760))

        # configure grid layout (4x4)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)


        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self.master, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Reinforcement Learning \n and \n Radiotherapy", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Back", width=190, command=self.back_page)
        self.sidebar_button_4.grid(row=1, column=0, padx=10, pady=10)

        
        
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], width=190,
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))

        self.tabview = customtkinter.CTkTabview(self.master)
        self.tabview.grid(row=0, column=1, rowspan=4, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.tabview.add("Reinforcement Learning")
        self.tabview.add("Radiotherapy")
        self.tabview.tab("Reinforcement Learning").grid_columnconfigure(0, weight=1)  
        self.tabview.tab("Radiotherapy").grid_columnconfigure(0, weight=1)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
    
    def back_page(self):
        self.app = App()