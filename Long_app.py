

import os
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pywt
from openpyxl import Workbook

fs = 12.5
TIME_PER_STEP = 1/fs

# Separate class to handle widget creation
class WidgetFactory:

    def create_label_frame(self, parent, bg_color, width, height, relx, rely):
        """Helper function to create a frame."""
        frame = tk.Frame(parent, bg=bg_color, width=width, height=height)
        frame.place(relx=relx, rely=rely)
        return frame

    def create_button(self, parent, text, command, relx, rely, width=None, height=None):
        """Helper function to create a button."""
        button = tk.Button(parent, text=text, command=command, width=width, height=height)
        button.place(relx=relx, rely=rely)
        return button

    def create_slider(self, parent, relx, rely, command):
        """Helper function to create a slider."""
        slider = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, command=command, width=10, length=500)
        slider.place(relx=relx, rely=rely)
        return slider

    def create_label(self, parent, text, relx, rely, width=None, height=None, transparent=False, color=False, words=False):
        """Helper function to create a label."""
        
        if transparent:
            label = tk.Label(parent, text=text, width=width, height=height,bg=color, borderwidth=0, highlightthickness=0, fg=words)
        else:
            label = tk.Label(parent, text=text, width=width, height=height)
        label.place(relx=relx, rely=rely)
        return label

    def create_combobox(self, parent, values, start, relx, rely, width=None, height=None):
        """Helper function to create a combobox."""
        combobox = ttk.Combobox(parent, values=values, width=width, height=height)
        combobox.place(relx=relx, rely=rely)
        combobox.set(start)
        return combobox

    def create_entry(self, parent, relx, rely):
        """Helper function to create an entry."""
        entry = tk.Entry(parent)
        entry.place(relx=relx, rely=rely)
        return entry


class Processor:
    
    # This function takes a csv file in input ant turns it into an array
    def csv_to_array(self, file_path):
        df = pd.read_csv(file_path)
        array = df.to_numpy()  # or df.values
        print(f"Loaded {os.path.basename(file_path)} with shape {array.shape}")
        return array
    
    def signal_eval (self, AP, CC, ML, t1, t2, t3):
        step = 48
        step_len = 24
        
        label = []
        
        for j in range(0,len(AP)-step_len,step_len):
            
            signal_AP = AP[j:j+step]       
            signal_CC = CC[j:j+step]       
            signal_ML = ML[j:j+step]
            
            x, _ = self.waveletter(signal_AP)
            high_AP = max(x[89:99])
            AP_max = np.max(np.abs(signal_AP))
            CC_max = np.max(np.abs(signal_CC))
            ML_max = np.max(np.abs(signal_ML))
            
            l = self.act_identifier(high_AP,AP_max,CC_max,ML_max,t1,t2,t3)
            
            label.append(l)
        
        label=np.array(label)
        label = np.repeat(label,24)
        
        return label
                     
    def act_identifier (self,h_AP, AP_max, CC_max, ML_max, t1, t2, t3):

        if ML_max > 2.5 and CC_max> 4 and AP_max<2.9:
            act = 0
        elif CC_max < t1:
            act = 1
        elif h_AP < t2:
            act = 2
        elif h_AP < t3:
            act = 3
        else:
            act = 4

        return act
    
    def waveletter (self, data):
        S = 100
        scales = np.arange(1,S)
        wavelet = 'cmor1.0-0.5'

        coeffs, freqs = pywt.cwt(data, scales, wavelet)
        mean_coeffs = np.zeros(len(coeffs[:,0]))
        for j in range (len (mean_coeffs)):
            mean_coeffs[j] = np.mean(abs(coeffs[j,:]))
        x = mean_coeffs[::-1]       
        r = np.sqrt(np.mean(np.square(data)))
        return x, r
    
    def format_duration(self,seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
    
    def stepcounter(self, data, thresh, flag=0):
        signal = np.ravel(np.array(data, dtype=float))  # Ensure it's a flat NumPy array
        t = float(thresh)  # Also ensure the threshold is float

        minima ,_ = find_peaks(-signal)
        m = np.mean(signal[minima])
        maxima ,_ = find_peaks(signal)
        M = np.mean(signal[maxima])
        a = signal.shape
        signal = 2 * ((signal - np.ones(a) * m) / (M - m)) - 1

        signal[signal < 0] = 0
        pks, properties = find_peaks(signal, height=(np.mean(signal) * t))

        if flag == 0:
            steps = len(pks) / 2
        else:
            steps = len(pks)

        return steps


class PlotFrame:
    
    def __init__(self, parent):
        self.frame = tk.Frame(parent)
        self.frame.place(relx=0.03, rely=0.12, relwidth=0.6, relheight=0.84)
        self.figure = plt.Figure(figsize=(9.6, 8.4), dpi=50)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack()

    def plot_signal(self, signal_data, label="Signal"):
        """Method to plot the signal data."""
        self.ax.clear()  # Clear any previous plot
        self.ax.plot(signal_data, label=label)
        self.ax.set_title("Signal Plot")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.canvas.draw()
        
    def plot_activity_distribution(self, activity_labels, durations, format_duration_fn, title="Activity Distribution"):
        self.ax.clear()
    
        total = sum(durations)
        percentages = [(d / total) * 100 if total > 0 else 0 for d in durations]
        durations_fmt = [format_duration_fn(d) for d in durations]

        # Plot bars
        bars = self.ax.bar(activity_labels, percentages, color="lightblue")

        # Add formatted time on top of each bar
        for bar, label in zip(bars, durations_fmt):
            height = bar.get_height()
            self.ax.annotate(
                label,
               (bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom', fontsize=9
            )

        self.ax.set_title(title)
        self.ax.set_ylabel("Percentage (%)")
        self.ax.set_xlabel("Activity")
        self.canvas.draw()
    
    
    
    
# Main application class
class ElabApp:
    
    "Creation of the GUI"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Long Term Analysis App")
        self.root.geometry("800x500")
        self.root.configure(bg="#B4D3CE")
        
        # Instantiate the WidgetFactory to use for creating widgets
        self.widget_factory = WidgetFactory()
        self.processor = Processor() 
        self.plot_frame = PlotFrame(self.root)
        
        self.create_widgets()
        self.initialize_variables()
    
    def create_widgets(self):
        
        self.control_frame = self.widget_factory.create_label_frame(self.root, "#698b94", 800, 40, 0, 0)
        self.thresholds_frame = self.widget_factory.create_label_frame(self.root, "#698b94", 250, 250, 0.66, 0.12)
        self.stepcoutning_frame = self.widget_factory.create_label_frame(self.root, "#698b94", 250, 150, 0.66, 0.66)
        
        self.plot_frame = PlotFrame(self.root)
        
        # Create and place buttons, sliders, and labels using the WidgetFactory methods
        self.load_button = self.widget_factory.create_button(self.root, "Load Signal", self.load_signal, 0.01, 0.015)
        self.activity_eval_button = self.widget_factory.create_button(self.root, "Activity Eval", self.activity_eval, 0.12, 0.015)
        self.time_eval_button = self.widget_factory.create_button(self.root, "Time Eval", self.time_eval, 0.23, 0.015)
        self.save_button = self.widget_factory.create_button(self.root, "Save", self.save, 0.95, 0.015)
        
        self.title_label = self.widget_factory.create_label(self.root, "Activity label thresholds ", 0.68, 0.14, 0, 0, 
                                                            transparent = 1, color = "#698b94", words='white')
        self.t1_label = self.widget_factory.create_label(self.root, "Thresh label 1: ", 0.68, 0.19, 11, 1)
        self.t2_label = self.widget_factory.create_label(self.root, "Thresh label 2: ", 0.68, 0.25, 11, 1)
        self.t3_label = self.widget_factory.create_label(self.root, "Thresh label 3: ", 0.68, 0.31, 11, 1)
        
        self.t1_label_entry = self.widget_factory.create_entry(self.root, 0.8, 0.19)
        self.t2_label_entry = self.widget_factory.create_entry(self.root, 0.8, 0.25)
        self.t3_label_entry = self.widget_factory.create_entry(self.root, 0.8, 0.31)
        
        self.title_steps = self.widget_factory.create_label(self.root, "Stepcounting thresholds ", 0.68, 0.38, 0, 0, 
                                                            transparent = 1, color = "#698b94", words='white')
        self.t1_steps = self.widget_factory.create_label(self.root, "Thresh steps 1: ", 0.68, 0.43, 11, 1)
        self.t2_steps = self.widget_factory.create_label(self.root, "Thresh steps 2: ", 0.68, 0.49, 11, 1)
        self.t3_steps = self.widget_factory.create_label(self.root, "Thresh steps 3: ", 0.68, 0.55, 11, 1)
        
        self.t1_steps_entry = self.widget_factory.create_entry(self.root, 0.8, 0.43)
        self.t2_steps_entry = self.widget_factory.create_entry(self.root, 0.8, 0.49)
        self.t3_steps_entry = self.widget_factory.create_entry(self.root, 0.8, 0.55)
        
        self.title_count = self.widget_factory.create_label(self.root, "Stepcounting elaboration ", 0.68, 0.68, 0, 0, 
                                                            transparent = 1, color = "#698b94", words='white')
        self.walk_counting_button = self.widget_factory.create_button(self.root, "Walk", self.walk_count, 0.68, 0.73, width=5)
        self.jog_counting_button = self.widget_factory.create_button(self.root, "Jog", self.jog_count, 0.68, 0.81, width=5)
        self.sprint_counting_button = self.widget_factory.create_button(self.root, "Sprint", self.sprint_count, 0.68, 0.89, width=5)
        self.walk_count = self.widget_factory.create_label(self.root, "Steps: ", 0.75, 0.74, 0, 0, 
                                                            transparent = 1, color = "#698b94", words='white')
        self.jog_count = self.widget_factory.create_label(self.root, "Steps: ", 0.75, 0.82, 0, 0, 
                                                            transparent = 1, color = "#698b94", words='white')
        self.sprint_count = self.widget_factory.create_label(self.root, "Steps: ", 0.75, 0.9, 0, 0, 
                                                            transparent = 1, color = "#698b94", words='white')
    
    def initialize_variables(self):
        
        self.walk = []
        self.jog = []
        self.sprint = []
        self.label = []
        
        self.time_stop = 0
        self.time_walk = 0
        self.time_jog = 0
        self.time_sprint = 0
        self.time_other = 0
        
        self.activity_eval_button.config(state="disabled")
        self.time_eval_button.config(state="disabled")
        
        self.walk_counting_button.config(state="disabled")
        self.jog_counting_button.config(state="disabled")
        self.sprint_counting_button.config(state="disabled")
        self.save_button.config(state="disabled")
        
    # Functions for the widgets

    def load_signal(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        
        if not folder_path:
            print("No folder selected.")
            return
        
        #Create a folder for the output
        output_folder = os.path.join(folder_path, "Output")
        os.makedirs(output_folder, exist_ok=True)
        
        # List all CSV files in the selected folder
        self.csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        
        if not self.csv_files:
            print("No CSV files found in the selected folder.")
            return
        self.activity_eval_button.config(state="normal")
    
    def activity_eval(self):
        
        # Convert all the csv files into arrays
        # Then extract the part of the signals that correspond to
        # a specific activity
        
        for file_path in self.csv_files:
            try:
                t1 = float(self.t1_label_entry.get())
                t2 = float(self.t2_label_entry.get())
                t3 = float(self.t3_label_entry.get())
                array = self.processor.csv_to_array(file_path)
                label_iter = self.processor.signal_eval(array[:,0],array[:,1],array[:,2],
                                                   t1,t2,t3)
                
                idx = np.where(label_iter == 2)
                self.walk.append(array[idx,0])
                idx = np.where(label_iter == 3)
                self.jog.append(array[idx,0])
                idx = np.where(label_iter == 4)
                self.sprint.append(array[idx,0])
                self.label.append(label_iter)
                
                
                #print(f'{self.label}')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Concatenate the signals into a single array
        # thenChek if the signal for every activity is empty, and
        # activate the available stepcounting buttons
        
        if self.walk:
            combined = np.concatenate(self.walk).flatten()
            if combined.size > 0:
                self.walk = combined
                self.walk_counting_button.config(state="normal")
                #print(f'{self.walk}, shape: {self.walk.shape}')
                #self.plot_frame.plot_signal(self.walk)
        
        if self.jog:
            combined = np.concatenate(self.jog).flatten()
            if combined.size > 0:
                self.jog = combined
                self.jog_counting_button.config(state="normal")
            
        if self.sprint:
            combined = np.concatenate(self.sprint).flatten()
            if combined.size > 0:
                self.sprint = combined
                self.sprint_counting_button.config(state="normal")
                # print(f'{self.sprint}, shape: {self.sprint.shape}')
                # self.plot_frame.plot_signal(self.sprint)
            
        if len(self.label) > 0:
            self.label = np.concatenate(self.label).flatten()
        
        self.time_eval_button.config(state="normal")
    
    def time_eval(self):
        
        self.time_stop += (np.sum(self.label == 1)/fs) #/60
        self.time_walk += (np.sum(self.label == 2)/fs) #/60
        self.time_jog += (np.sum(self.label == 3)/fs) #/60
        self.time_sprint += (np.sum(self.label == 4)/fs) #/60
        self.time_other += (np.sum(self.label == 0)/fs) #/60

        print(f"Stopped: {self.processor.format_duration(self.time_stop)}, "
          f"Walk: {self.processor.format_duration(self.time_walk)}, "
          f"Jog: {self.processor.format_duration(self.time_jog)}, "
          f"Sprint: {self.processor.format_duration(self.time_sprint)}, "
          f"Other: {self.processor.format_duration(self.time_other)}")
        
        activity_labels = ["Walk", "Jog", "Sprint", "Other"]
        durations = [
            self.time_walk,
            self.time_jog,
            self.time_sprint,
            self.time_other
            ]
        
        # Assume self.plotter is an instance of PlotFrame
        self.plot_frame.plot_activity_distribution(
        activity_labels,
        durations,
        self.processor.format_duration,  # your formatter
        title="Activity Distribution"
        )
        
        self.save_button.config(state="normal")
    
    def walk_count(self):
        self.walk_steps = self.processor.stepcounter(self.walk, 
                                           self.t1_steps_entry.get())
        self.walk_count = self.widget_factory.create_label(self.root, f"Steps: {self.walk_steps}", 
                                                           0.75, 0.74, 0, 0, 
                                                           transparent = 1, color = "#698b94", words='white')
        
    def jog_count(self):
        self.jog_steps = self.processor.stepcounter(self.jog, 
                                           self.t2_steps_entry.get())
        self.jog_count = self.widget_factory.create_label(self.root, f"Steps: {self.jog_steps}", 
                                                           0.75, 0.82, 0, 0,   
                                                           transparent = 1, color = "#698b94", words='white')
        
    def sprint_count(self):
        self.sprint_steps = self.processor.stepcounter(self.sprint, 
                                           self.t3_steps_entry.get(),1)
        self.sprint_count = self.widget_factory.create_label(self.root, f"Steps: {self.sprint_steps}", 
                                                           0.75, 0.9, 0, 0, 
                                                           transparent = 1, color = "#698b94", words='white')
    
    def save(self):
        
        activity_sequence = []
        current_activity = self.label[0]
        start_idx = 0

        for i in range(1, len(self.label)):
            if self.label[i] != current_activity:
                duration_sec = (i - start_idx) / fs
                duration_str = self.processor.format_duration(duration_sec)
                activity_sequence.append(f"{int(current_activity)}({duration_str})")
                current_activity = self.label[i]
                start_idx = i

        # Ultimo blocco
        duration_sec = (len(self.label) - start_idx) /fs
        duration_str = self.processor.format_duration(duration_sec)
        activity_sequence.append(f"{int(current_activity)}({duration_str})")


        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"

        # Header
        ws.append(["Activity", "Duration", "Steps", "Sequence"])

        durations = [
            self.time_other,  # Activity 0
            self.time_stop,   # Activity 1
            self.time_walk,   # Activity 2
            self.time_jog,    # Activity 3
            self.time_sprint  # Activity 4
        ]

        step_counts = [
            "",  # Activity 0 - Other
            "",  # Activity 1 - Stop
            getattr(self, "walk_steps", ""),   # Activity 2 - Walk
            getattr(self, "jog_steps", ""),    # Activity 3 - Jog
            getattr(self, "sprint_steps", "")  # Activity 4 - Sprint
        ]

        # Fill the first 5 rows with Activity, Duration, Step count, and Sequence
        for i in range(5):
            ws.append([
                i,  # Activity column (or Activity ID)
                self.processor.format_duration(durations[i]),  # Duration column
                step_counts[i],  # Step count column
                activity_sequence[i] if i < len(activity_sequence) else ""  # Sequence column
            ])

        # After that, we continue to populate the sequence column with remaining activities
        for idx in range(5, len(activity_sequence)):
            # Append only the activity to the sequence column
            # Assuming we're filling the 4th column (Sequence)
            ws.append(["", "", "", activity_sequence[idx]])  # Only sequence column gets populated
            
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel files", "*.xlsx")],
                                                title="Save Summary As")
        if save_path:
            wb.save(save_path)
            print(f"Summary saved to {save_path}")
        else:
            print("Save cancelled.")
        


        
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ElabApp(root)
    root.mainloop()
