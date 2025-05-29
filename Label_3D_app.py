import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging


# Color Palette
BG_color = "#B4D3CE"
frame_color = "#698b94"


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

    def create_label(self, parent, text: str, relx: float, rely: float, width: int = None, height: int = None, transparent: bool = False, color = False, words = False):
        
        """
        Helper function to create a label.
    
        Args:
            parent: Parent tkinter widget.
            text (str): The text to display.
            relx (float): Relative X position.
            rely (float): Relative Y position.
            width (int, optional): Width of the label.
            height (int, optional): Height of the label.
            transparent (bool, optional): If True, allows for the color to be modified
            bg_color (str, optional): Background color (used if transparent=True).
            fg_color (str, optional): Foreground (text) color.

        Returns:
            tk.Label: The created label widget.
        """
        
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
    
class VideoFrame:
    def __init__(self, root):
        self.frame = tk.Frame(root)
        self.frame.place(relx=0.02, rely=0.1)
        self.canvas = tk.Canvas(self.frame, width=500, height=400, bg="black")
        self.canvas.pack()
        
        
    def display_frame(self, frame):
        """Displays the given frame on the canvas."""
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Keep reference to the image to prevent garbage collection

    def clear(self):
        """Clear the canvas."""
        self.canvas.delete("all")

class AccPlotFrame:
    def __init__(self, parent):
        self.frame = tk.Frame(parent)
        self.frame.place(relx=0.46, rely=0.1, relwidth=0.25, relheight=0.85)
        self.figure = plt.Figure(figsize=(6, 10.2), dpi=50)
        
        self.ax = [
            self.figure.add_subplot(311),
            self.figure.add_subplot(312),
            self.figure.add_subplot(313)
        ]
        
        titles = ['Antero-Posterior', 'Cranio-Caudal', 'Medio-Lateral']
        
        for i in range (3):
            # Configure the subplots
            self.ax[i].set_xlim(0, 100)
            self.ax[i].set_ylim(-1, 1)
            self.ax[i].set_title(titles[i])
            self.ax[i].set_xlabel('Frames')
            self.ax[i].set_ylabel('Acceleration')
        
        self.figure.subplots_adjust(hspace=0.3)
        self.figure.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack()

        # Initialize line and marker as None
        self.line = None
        self.marker = None
    
    def plot_on_subplot(self, data1, data2, data3, xmax, xmin):
        
        self.clear()
        l = len(data1)
        self.ax[0].plot(data1, color='blue')
        self.ax[0].set_xlim(0, l)
        self.ax[0].set_ylim(xmin[0], xmax[0])
        self.ax[1].plot(data2, color='green')
        self.ax[1].set_xlim(0, l)
        self.ax[1].set_ylim(xmin[1], xmax[1])
        self.ax[2].plot(data3, color='black')
        self.ax[2].set_xlim(0, l)
        self.ax[2].set_ylim(xmin[2], xmax[2])
        print("l",l)
        self.canvas.draw()  # Update the canvas to display the plot

    
    def plot_acceleration(self, data1, data2, data3, frame_count, xmax, xmin):
        self.clear()
        frame_max = len(data1)
        if frame_count < 0 or frame_count >= frame_max:
            print(f"Warning: Frame {frame_count} is out of bounds.")
            return

        # Determine the interval range
        x1 = max(0, frame_count - 50)  # Show 50 frames before
        x2 = min(frame_max, frame_count + 50)  # Show 50 frames after
        x_values = range(x1, x2)  # X-axis values for the interval

        # Update line (replot only the interval data)
        self.line, = self.ax[0].plot(x_values, data1[x1:x2], color='blue')
        self.marker = self.ax[0].scatter(frame_count, data1[frame_count], color='red', zorder=5)
        # Adjust axis limits
        self.ax[0].set_xlim(x1, x2)
        self.ax[0].set_ylim(xmin[0], xmax[0])  # Set Y-axis to fit the data in the interval
        
        # Update line (replot only the interval data)
        self.line, = self.ax[1].plot(x_values, data2[x1:x2], color='green')
        self.marker = self.ax[1].scatter(frame_count, data2[frame_count], color='red', zorder=5)
        # Adjust axis limits
        self.ax[1].set_xlim(x1, x2)
        self.ax[1].set_ylim(xmin[1], xmax[1])  # Set Y-axis to fit the data in the interval
        
        # Update line (replot only the interval data)
        self.line, = self.ax[2].plot(x_values, data3[x1:x2], color='black')
        self.marker = self.ax[2].scatter(frame_count, data3[frame_count], color='red', zorder=5)
        # Adjust axis limits
        self.ax[2].set_xlim(x1, x2)
        self.ax[2].set_ylim(xmin[2], xmax[2])  # Set Y-axis to fit the data in the interval
        
        # Redraw the canvas
        self.canvas.draw()
        
    def clear(self):
        """Clear the plot data but retain the title and axis labels."""
        for ax in self.ax:
            # Clear only the data (lines and markers)
            for line in ax.lines:
                line.set_data([], [])  # Remove data for each line
            for collection in ax.collections:
                collection.remove()  # Remove any markers
            
    def plot(self, data, color='blue'):
        """Plot the data."""
        self.ax.plot(data, color=color)
        self.canvas.draw()  # Update the canvas with the new plot

    def set_title(self, title):
        self.ax.set_title(title)
        self.canvas.draw()

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)
        self.canvas.draw()

    def set_ylabel(self, label):
        self.ax.set_ylabel(label)
        self.canvas.draw()

    def draw(self):
        self.canvas.draw() 
 
    
 
    
class VideoApp:
    
    "Creation of the GUI"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Labeling App")
        self.root.geometry("1200x600")
        self.root.configure(bg = BG_color)
        
        self.widget_factory = WidgetFactory()
        self.create_widgets()
        self.initialize_variables()
        self.initialize_buttons()

    def create_widgets(self):
        
        """Create and place all GUI widgets including frames, buttons, labels, sliders, and input controls."""
        
        self.label_frame = self.widget_factory.create_label_frame(self.root, frame_color, 1200, 40, 0, 0)
        self.control_frame = self.widget_factory.create_label_frame(self.root, frame_color, 300, 200, 0.73, 0.1)
        
        self.video_frame = VideoFrame(self.root)
        self.acc_frame = AccPlotFrame(self.root)
        
        # Create and place buttons
        self.load_button = self.widget_factory.create_button(self.root, "Load Video", self.load_video, 0.01, 0.01)
        self.load_acc_button = self.widget_factory.create_button(self.root, "Load Acc", self.load_acc, 0.09, 0.01)
        self.play_button = self.widget_factory.create_button(self.root, "Play", self.start_video, 0.02, 0.9)
        self.stop_button = self.widget_factory.create_button(self.root, "Stop", self.stop_video, 0.09, 0.9)
        self.fast_forward_button = self.widget_factory.create_button(self.root, ">>", self.fast_forward, 0.29, 0.9)
        self.rewind_button = self.widget_factory.create_button(self.root, "<<", self.rewind, 0.22, 0.9)
        self.set_start_frame_button = self.widget_factory.create_button(self.root, "Start", self.set_start_frame, 0.74, 0.12, 8, 1)
        self.set_end_frame_button = self.widget_factory.create_button(self.root, "End", self.set_end_frame, 0.74, 0.17, 8, 1)
        self.save_button = self.widget_factory.create_button(self.root, "Save", self.save_data, 0.74, 0.37, 8, 1)
        self.combine_button = self.widget_factory.create_button(self.root, "Combine", self.combine, 0.81, 0.37, 8, 1)
        self.save_combine_button = self.widget_factory.create_button(self.root, "Save comb", self.save_combine, 0.88, 0.37, 8, 1)
        
        # Sliders and Labels
        self.slider = self.widget_factory.create_slider(self.root, 0.02, 0.8, self.slider_update)
        self.frame_number_label = self.widget_factory.create_label(self.root, "Frame: 0", 0.34, 0.9)
        self.Hz_label = self.widget_factory.create_label(self.root, "Hz:", 0.18, 0.01)
        self.start_frame_label = self.widget_factory.create_label(self.root, "Start Frame: ", 0.809, 0.125, transparent = 1, color = "#698b94", words = 'white')
        self.end_frame_label = self.widget_factory.create_label(self.root, "End Frame: ", 0.809, 0.175, transparent = 1, color = "#698b94", words = 'white')
        self.activity_label = self.widget_factory.create_label(self.root, "  Activity: ", 0.74, 0.22, 8, 1)
        self.steps_label = self.widget_factory.create_label(self.root, " # Steps: ", 0.74, 0.27, 8, 1)
        self.sbj_label = self.widget_factory.create_label(self.root, " Sbj: ", 0.74, 0.32, 8, 1)
        self.speed_label = self.widget_factory.create_label(self.root, "Speed", 0.052, 0.88, 5, 1, transparent = 1, color = "#B4D3CE", words = 'black')
        self.skip_label = self.widget_factory.create_label(self.root, "Skip:", 0.25, 0.88, 5, 1, transparent = 1, color = "#B4D3CE", words = 'black')
        
        # Additional controls (comboboxes, text entries, etc.)
        self.Hz_combobox = self.widget_factory.create_combobox(self.root, ["800", "400", "200", "100", "50", "25", "12.5"], "12.5", 0.21, 0.01, 5)
        self.Activity_combobox = self.widget_factory.create_combobox(self.root, ["Other", "Stop", "Walk", "Jog", "Sprint", "Transition"], "Other", 0.81, 0.22)
        self.vel_combobox = self.widget_factory.create_combobox(self.root, ["2","4","8","16"], "4",  0.05, 0.91, 3)
        self.skip_combobox = self.widget_factory.create_combobox(self.root, ["1","5","10","50","100","200","500","1000"], "10",  0.25, 0.91, 3)     
        self.steps_entry = self.widget_factory.create_entry(self.root, 0.81, 0.27);
        self.sbj_entry = self.widget_factory.create_entry(self.root, 0.81, 0.32);

    def initialize_variables(self):
        
        """Initialize class variables."""
        
        self.vid = None
        self.is_playing = False
        self.xs = None  # This will hold the IMU data
        self.start_frame = None;
        self.end_frame = None;
        self.movement_type = None;
        self.num_steps = None;
        
        self.activity_mapping = {
        "Other": 0,
        "Stop": 1,
        "Walk": 2,
        "Jog": 3,
        "Sprint": 4,
        "Transition": 5
        }

    def initialize_buttons(self):
        
        """Initialize button states."""
        
        self.load_button.config(state="normal")
        self.load_acc_button.config(state="disabled")
        self.play_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.fast_forward_button.config(state="disabled")
        self.rewind_button.config(state="disabled")
        
    "Functions for video and IMU display"
    
    def load_video(self):
        
        # This function is connected to the load video button
        # It allows the user to choose a video file, then loads it in the
        # app, updates the slider to match the length of the video and 
        # displays the first frame of the video
        
        file_path = filedialog.askopenfilename(filetypes=[("All Video files", "*.*")])
        
        if not file_path:
            messagebox.showinfo("No File Selected", "Please select a file to proceed.")
            return
        
        self.vid = cv2.VideoCapture(file_path)
        if not self.vid.isOpened():
            messagebox.showerror("Load Error", "The selected file could not be opened as a video.")
            return
    
        # Display first frame
        ret, frame = self.vid.read()
        if not ret:
            messagebox.showerror("Read Error", "Unable to read the first frame of the video.")
            return
        
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (500, 400))
            self.video_frame.display_frame(frame)  # Use video_frame to display the frame
        except Exception as e:
            messagebox.showerror("Display Error", f"An error occurred while displaying the video frame:\n{e}")
            return
        
        # Update slider properties after video is loaded
        self.slider.config(to=int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.load_button.config(state="disabled")
        self.load_acc_button.config(state="normal")

    def load_acc(self):
        
        # This function is connected to the load acc button
        # It allows the user to choose a csv file, then loads it in the
        # app, resamples the signal so it matches the video and plots in
        # the three diagrams the whole signal, so that the user can
        # evaluate potential problems before the labeling starts
        
        freq_map = {'800': 64, '400': 32, '200': 16, '100': 8, '50': 4, '25': 2, '12.5' : 1}

        # Open file dialog to load acceleration data
        acc_file = filedialog.askopenfilename(filetypes=[("Acceleration data files", "*.csv")])
        if not acc_file:
            messagebox.showinfo("No File Selected", "Please select a file to proceed.")
            return

        # Read the data
        try:
            df = pd.read_csv(acc_file)
        except Exception as e:
            messagebox.showerror("File Read Error", f"An error occurred while reading the file:\n{e}")
            return
        
        # Validate that the file has enough columns
        required_cols = 4  # Assuming at least 4 columns including index or time
        if df.shape[1] < required_cols:
            messagebox.showerror("Invalid File Format", f"The file must contain at least {required_cols} columns.")
            return
        
        try: 
            C_vert, C_hor, C_tra = 1, 2, 3
            x = df.iloc[:, [C_vert, C_hor, C_tra]].to_numpy()
        
            self.xmax = [np.max(x[:,0]),np.max(x[:,1]),np.max(x[:,2])]
            self.xmin = [np.min(x[:,0]),np.min(x[:,1]),np.min(x[:,2])]
        
            # Get selected frequency from combobox
            selected_freq = self.Hz_combobox.get()
            resample_factor = freq_map.get(selected_freq)
        
            if selected_freq == '12.5':
                # For 12.5Hz resampling, repeat each value twice
                self.xs = np.zeros((len(x[:, 0]),3))
                self.xs[:, 0] = x[:, 0].flatten() 
                self.xs[:, 1] = x[:, 1].flatten() 
                self.xs[:, 2] = x[:, 2].flatten() 
            else:
                # Resampling based on selected frequency
                resampled_len = int(len(x[:, 0])/resample_factor)
                xs1 = np.zeros((resampled_len,1))
                xs2 = np.zeros((resampled_len,1))
                xs3 = np.zeros((resampled_len,1))
                cont =0
                for i in range (resampled_len):          
                    xs1[i] = x[cont, 0]
                    xs2[i] = x[cont, 1]
                    xs3[i] = x[cont, 2]
                    cont = cont + resample_factor
                self.xs = np.zeros((resampled_len,3))
                self.xs[:,0] = xs1.flatten() 
                self.xs[:,1] = xs2.flatten() 
                self.xs[:,2] = xs3.flatten() 

            # Now manually plot each column (AP, ML, TR) on its respective subplot
            self.acc_frame.plot_on_subplot(self.xs[:,0],self.xs[:,1],self.xs[:,2],self.xmax,self.xmin)
            self.load_acc_button.config(state="disabled")
            self.play_button.config(state="normal")
            self.fast_forward_button.config(state="normal")
            self.rewind_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred while processing the data:\n{e}")
        
    def start_video(self):
        
        """Start video playback at the selected speed."""
        
        try:
            self.vel = int(self.vel_combobox.get())
        except ValueError:
            messagebox.showerror("Invalid Speed", "Please select a valid playback speed.")
            return
        
        if self.vid:
            self.is_playing = True;
            self.play_video();
            self.play_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.fast_forward_button.config(state="disabled")
            self.rewind_button.config(state="disabled")
        else:
            messagebox.showerror("No Video", "Please load a valid video before playing.")

    def stop_video(self):
        
        ''' Stop the video '''
        
        if self.vid:
            self.is_playing = False;
            self.play_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.fast_forward_button.config(state="normal")
            self.rewind_button.config(state="normal")
        else:
            messagebox.showerror("No Video", "Please load a valid video before playing.")

    def fast_forward(self):
        
        """ Advance the video by a user-defined number of frames."""
        
        try:
            skip = int(self.skip_combobox.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please select a valid skip value.")
            return
        
        if self.vid:
            current_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
            target_frame = min(current_frame + skip - 2, self.vid.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            self.process_and_display_frame(target_frame)

    def rewind(self):
        
        """ Rewind the video by a user-defined number of frames."""
        
        try:
            skip = int(self.skip_combobox.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please select a valid skip value.")
            return
        
        if self.vid:
            current_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
            target_frame = max(current_frame - skip - 2, 0) 
            self.process_and_display_frame(target_frame)
                
    def play_video(self):
        
        ''' Video Playing Loop '''
        
        if not self.is_playing:
            return
        
        frame_count = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number
        frame_max = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
        self.process_and_display_frame(frame_count)
            
        if frame_count < frame_max - 1:  # Continue playing if there are more frames
            self.root.after(1, self.play_video) #The numvber defines the video displaying speed
        else:
            # End of video reached, rewind to the beginning
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_video()
                       
    def slider_update(self, value):
        
        ''' Function connected to the slider '''
        
        if not self.vid or self.is_playing:
            return
        try: 
            frame_count = int(value)
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = self.vid.read()
            if not ret:
                return 
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (500, 400));
            self.video_frame.display_frame(frame)
            self.acc_frame.plot_acceleration(self.xs[:,0], self.xs[:,1], self.xs[:,2], int(round(frame_count/2)), self.xmax, self.xmin)
            self.frame_number_label.config(text="Frame: " + str(round(frame_count/2)));
        except (ValueError, TypeError) as e:
            # Handle cases where `value` isn't a valid number
            print(f"Invalid frame value: {value} ({e})")
            
    def process_and_display_frame(self, frame_number):
        
        ''' Function to process and display the needed frame '''
        
        if not self.vid:
            return
        
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.vid.read()
        if not ret:
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (500, 400))
        self.video_frame.display_frame(frame)
        
        frame_count = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_count % self.vel == 0:
            self.acc_frame.plot_acceleration(self.xs[:,0], self.xs[:,1], self.xs[:,2], int(frame_count/2), self.xmax, self.xmin)
        self.frame_number_label.config(text="Frame: " + str(round(frame_count/2)))
        self.slider.set(frame_count)
        
    "Saving Functions"
    
    def update_num(self, event):
        
        """
        Updates the 'num' variable based on the currently selected activity
        from the combobox using a predefined mapping.
        """
    
        # Get the selected activity
        selected_activity = self.activity_options[self.activity_combobox.current()]

        # Look up the corresponding number from the mapping
        selected_number = self.activity_mapping.get(selected_activity)

        # Update the value of num
        if selected_number is not None:
            self.num.set(selected_number)
            
    def set_start_frame(self):
        
        """
        Sets the starting frame as half of the current position in the
        video, which corresponds to the position in the acc file.
        Updates the start frame label and logs the action.
        """
    
        if self.vid is not None and self.vid.isOpened():
            self.start_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES)/2);
            self.start_frame_label.config(text="Start Frame: " + str(self.start_frame));
            
            # Log the start frame for debugging/record purposes
            logging.info(f"Start frame set to: {self.start_frame}")
        
        else:
            logging.error("Failed to set start frame: Video not loaded properly.")
            
    def set_end_frame(self):
        
        """
        Sets the ending frame as half of the current position in the
        video, which corresponds to the position in the acc file.
        Updates the end frame label and logs the action.
        """
        
        if self.vid is not None and self.vid.isOpened():
            self.end_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES)/2);
            self.end_frame_label.config(text="End Frame: " + str(self.end_frame));
            
            # Log the start frame for debugging/record purposes
            logging.info(f"End frame set to: {self.end_frame}")
            
        else:
            logging.error("Failed to set end frame: Video not loaded properly.")
               
    def save_data(self):
        
        """
        Saves a selected interval of sensor data to a CSV file with metadata.
        """
    
        try:
            steps_digit = int(self.steps_entry.get())
            sbj = int(self.sbj_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Steps and Subject ID must be integers.")
            return
        
        # Get the selected activity from the combobox and map it to its numeric value
        selected_activity = self.Activity_combobox.get()
        activity_type = self.activity_mapping.get(selected_activity, 0)  # Default to 'Other' if not found
        
        # Get the starting and endig frames to delineate the interval of interest
        try:
            start_idx = int(self.start_frame)
            end_idx = int(self.end_frame)
            if start_idx >= end_idx:
                messagebox.showerror("Frame Range Error", "Start frame must be less than end frame.")
                return
        except (TypeError, ValueError):
            messagebox.showerror("Frame Error", "Invalid frame indices.")
            return
        
        x = self.xs[start_idx:end_idx,:]
        s = len(x[:,0])
        
        data = np.zeros((s,4))
        data[0,3] = activity_type
        data[1,3] = steps_digit
        data[:,0:3] = x

        csvfile = pd.DataFrame(data)

        # Generate a default filename
        default_filename = f"interval_{start_idx}_{end_idx}_{activity_type}_{sbj}.csv"

        # Ask the user to choose the save location and filename
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename,
            title="Choose directory and filename"
        )

        # Save the file if a path is selected
        if file_path:
            try:
                csvfile.to_csv(file_path, index=False, header=False)  # Save without row index
                logging.info(f"Data saved to: {file_path}")
            except Exception as e:
                logging.error(f"Error saving file: {e}")
                messagebox.showerror("Save Error", f"Could not save file:\n{e}")
        else:
            logging.info("Save operation cancelled.")
    
    # The following functions substitue the previous New_combiner app
    def combine(self):
        
        '''
        Combines CSV files from a selected folder into a single NumPy array.
        '''
        
        # Open a dialog to select the folder
        folder_path = filedialog.askdirectory(title="Select Folder with CSV Files")

        if not folder_path:
            logging.info("No folder selected for combination.")
            return
            logging.info(f"Selected folder: {folder_path}")  # Print the selected folder path
        
        # Initialize a list to store the arrays
        self.combined_data = []

        # Loop through all files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                logging.info(f"Processing file: {file_path}")  # Print the file being loaded
                
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path, header=None)
                    df=df.values
                    # Convert the DataFrame to a NumPy array and add to the list
                    signal = df
                    l=len(signal[:,0])
                    array = np.zeros((l,4))
                    array[:,0:3] = signal[:,0:3]
                    array[:,3] = np.ones(l)*df[0,3]
                        
                    self.combined_data.append(array)
                    
                except Exception as e:
                    logging.error(f"Failed to process file {file_name}: {e}")
            
            if self.combined_data:
                self.combined_array = np.vstack(self.combined_data)
                logging.info(f"Successfully combined {len(self.combined_data)} files.")
            else:
                messagebox.showwarning("No Data", "No valid CSV files were combined.")
    
    def save_combine(self):
        
        '''
        Saves the combined data array to a CSV file.
        Prompts the user to choose a location and filename.
        '''
        
        if not hasattr(self, 'combined_array') or self.combined_array is None:
            messagebox.showerror("Save Error", "No combined data available to save.")
            return
        
        try:
            sbj = int(self.sbj_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Subject ID must be integer.")
            return
        
        # Convert combined array to DataFrame and save to CSV
        dy = pd.DataFrame(self.combined_array)  # Specify delimiter
        default_filename = f"SBJ_{sbj}.csv"

        # Ask the user to choose the save location and filename
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename,
            title="Choose directory and filename"
        )

        # Save the file if a path is selected
        if file_path:
            try:
                dy.to_csv(file_path, index=False, header=False, sep=',')
                logging.info(f"Data saved to {file_path}")
                messagebox.showinfo("Success", f"File saved:\n{file_path}")
            except Exception as e:
                logging.error(f"Failed to save file: {e}")
                messagebox.showerror("Save Error", f"Failed to save file:\n{e}")
        else:
            logging.info("Save cancelled by user.")
        


                
if __name__ == "__main__":
    root = tk.Tk();
    app = VideoApp(root);
    root.mainloop();