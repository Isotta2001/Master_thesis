# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import shutil
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
    
class MonteCarloFunctions:
    
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    def Arrays_division(self, csv_files):
        """
        Divides the CSV files into different activity arrays based on their activity labels.
        It returns arrays and names for different activities.
        """
        # Initialize arrays for each activity and transition
        arrays_other = []
        arrays_stop = []
        arrays_walk = []
        arrays_jog = []
        arrays_run = []
        arrays_trans_acc = []
        arrays_trans_dec = []

        # Initialize lists for filenames corresponding to each array
        names_other = []
        names_stop = []
        names_walk = []
        names_jog = []
        names_run = []
        names_trans_acc = []
        names_trans_dec = []

        for file_path in csv_files:
            try:
                file_name = os.path.basename(file_path)
                df = pd.read_csv(file_path, header=None)
                df = df.values
                
                # Get the number of rows in the data
                l = len(df[:, 0])
                
                # Initialize the signal array to store data with activity labels
                signal = np.zeros((l, 5))
                act = df[0, 3]  # Activity label from the data

                # Copy data (first three columns) and set activity label (column 4)
                signal[:, 0:3] = df[:, 0:3]
                signal[0, 4] = df[1, 3]

                # Classify the signal data based on activity type (0: other, 1: stop, 2: walk, etc.)
                if act == 0:
                    signal[:, 3] = np.zeros(l)
                    arrays_other.append(signal)
                    names_other.append(file_name)
                elif act == 1:
                    signal[:, 3] = np.ones(l)
                    arrays_stop.append(signal)
                    names_stop.append(file_name)
                elif act == 2:
                    signal[:, 3] = np.ones(l) * 2
                    arrays_walk.append(signal)
                    names_walk.append(file_name)
                elif act == 3:
                    signal[:, 3] = np.ones(l) * 3
                    arrays_jog.append(signal)
                    names_jog.append(file_name)
                elif act == 4:
                    signal[:, 3] = np.ones(l) * 4
                    arrays_run.append(signal)
                    names_run.append(file_name)
                elif act == 5:
                    half = l // 2
                    mean_first_half = np.mean(np.abs(df[:half, 0]))
                    mean_second_half = np.mean(np.abs(df[half:, 0]))
                    if mean_first_half > mean_second_half:
                        signal[:, 3] = np.ones(l) * 5
                        arrays_trans_acc.append(signal)
                        names_trans_acc.append(file_name)
                    else:
                        signal[:, 3] = np.ones(l) * 6
                        arrays_trans_dec.append(signal)
                        names_trans_dec.append(file_name)
            
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        arrays = [arrays_other, arrays_stop, arrays_walk, arrays_jog, arrays_run]
        names = [names_other, names_stop, names_walk, names_jog, names_run]
        
        # Return arrays and names for transition activities (acceleration and deceleration)
        trans_acc = arrays_trans_acc[0] if arrays_trans_acc else []
        trans_dec = arrays_trans_dec[0] if arrays_trans_dec else []
        
        return arrays, names, trans_acc, trans_dec
    
    def rnd_positions(self, database, N):
        """
        Generates N random positions within the given activity database for test/training splits.
        """
        s = len(database[1])  # Number of 'stop' samples
        w = len(database[2])  # Number of 'walk' samples
        j = len(database[3])  # Number of 'jog' samples
        r = len(database[4])  # Number of 'run' samples
        
        poss = np.zeros((N, 4))
        for i in range(N):
            # Randomly select positions from each activity
            poss[i, 0] = random.randint(0, s - 1)
            poss[i, 1] = random.randint(0, w - 1)
            poss[i, 2] = random.randint(0, j - 1)
            poss[i, 3] = random.randint(0, r - 1)
        
        return poss
    
    def cut_and_save(self, database, poss, N, Test, Train, acc, dec):
        """
        Cuts the database into training and testing sets based on random positions and saves them to CSV files.
        """
        for i in range(N):
            # Copy the database for modification
            train = [lst.copy() for lst in database]
            train = [train[1], train[2], train[3], train[4]]  # Focus only on the main activities
            test = []
            
            # Create the test set by extracting the data at the randomized positions
            for j in range(4):
                test.append(train[j].pop(int(poss[i, j])))
            
            # Randomize the order of the test arrays
            w_test = test[1][0, 4]
            j_test = test[2][0, 4]
            r_test = test[3][0, 4]
            test[3] = np.vstack([dec, test[3], acc])
            random.shuffle(test)
            test = np.vstack(test)
            test[:, 4] = np.zeros((len(test[:, 4])))
            test[0, 4] = w_test
            test[1, 4] = j_test
            test[2, 4] = r_test
            
            # Save the test data
            test_file_path = os.path.join(Test, f"Test_{i}.csv")
            dy = pd.DataFrame(test)
            dy.to_csv(test_file_path, index=False, header=False)
            logging.info(f"Test data saved to: {test_file_path}")
            
            # Calculate the total number of steps for each activity remaining in the training data
            w_train = sum([x[0, 4] for x in train[1]])
            j_train = sum([x[0, 4] for x in train[2]])
            r_train = sum([x[0, 4] for x in train[3]])
            
            # Combine the training arrays into a single dataset
            for j in range(4):
                train[j] = np.vstack(train[j])
            train = np.vstack(train)
            train[:, 4] = np.zeros((len(train[:, 4])))
            train[0, 4] = w_train
            train[1, 4] = j_train
            train[2, 4] = r_train
            
            # Save the training data
            train_file_path = os.path.join(Train, f"Train_{i}.csv")
            dy = pd.DataFrame(train)
            dy.to_csv(train_file_path, index=False, header=False)
            logging.info(f"Train data saved to: {train_file_path}")
    
    
    
    
    
    
    
class MonteCarloApp:
    
    "Creation of the GUI"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Monte Carlo App")
        self.root.geometry("600x600")
        self.root.configure(bg = BG_color)
        
        self.widget_factory = WidgetFactory()
        self.create_widgets()
        self.monte_carlo = MonteCarloFunctions()
        
        
    def create_widgets(self):
        
        """Create and place all GUI widgets including frames, buttons, labels, sliders, and input controls."""
        #self.label_frame = self.widget_factory.create_label_frame(self.root, frame_color, 600, 40, 0, 0)
        
        self.folder_button = self.widget_factory.create_button(self.root, "Select Folder", self.select_folder, 0.01, 0.01)
        
        self.iterations_label = self.widget_factory.create_label(self.root, "# of iterations: ", 0.17, 0.015, 11, 1)
        self.iterations_label_entry = self.widget_factory.create_entry(self.root, 0.32, 0.015)
        
        self.database_button = self.widget_factory.create_button(self.root, "Create Database", self.database_create, 0.01, 0.07)
        self.clear_button = self.widget_factory.create_button(self.root, "Clear Database", self.clear_database, 0.32, 0.07)
    
    
    
    
    def select_folder(self):
        
        """
        Opens a folder selection dialog to choose a directory and verifies
        the existence of the folder and the required CSV files inside it.
        Creates 'Test' and 'Train' subdirectories if they do not exist.
        """
        
        try:
            # Prompt user to select a folder
            self.folder_path = filedialog.askdirectory(title="Select Folder")
            
            # If no folder is selected, display a message and return
            if not self.folder_path:
                logging.warning("No folder selected.")
                messagebox.showwarning("Folder Selection", "No folder was selected. Please select a folder to continue.")
                return

            # Create Test and Train directories if they don't exist
            self.test_folder = os.path.join(self.folder_path, "Test")
            self.train_folder = os.path.join(self.folder_path, "Train")
            
            os.makedirs(self.test_folder, exist_ok=True)
            os.makedirs(self.train_folder, exist_ok=True)
            logging.info(f"Test and Train directories created/verified: {self.test_folder}, {self.train_folder}")

            # List all CSV files in the selected folder
            self.csv_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.lower().endswith('.csv')]

            # If no CSV files are found, display a warning message
            if not self.csv_files:
                logging.warning("No CSV files found in the selected folder.")
                messagebox.showwarning("No CSV Files", "No CSV files were found in the selected folder. Please ensure the folder contains CSV files.")
                return

            logging.info(f"CSV files found: {len(self.csv_files)} files.")
            messagebox.showinfo("Folder Selected", f"Successfully selected folder: {self.folder_path}")
        
        except Exception as e:
            logging.error(f"Error selecting folder: {e}")
            messagebox.showerror("Error", f"An error occurred while selecting the folder:\n{e}")
    
    
    def database_create(self):
        
        """
        Creates a database for Monte Carlo simulation by generating training and test datasets.
        The function handles user input, validates the number of iterations, and processes CSV files.
        """
        
        try:
            # Get the number of iterations from the user input
            n = self.iterations_label_entry.get()
            
            if not n:
                messagebox.showerror("Input Error", "Number of iterations is required. Please enter a valid number of iterations.")
                logging.warning("No iterations selected.")
                return

            # Attempt to convert the input to an integer
            try:
                n = int(n)
                if n <= 0:
                    raise ValueError("Iterations must be a positive integer.")
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter a valid integer for the number of iterations.\nError: {e}")
                logging.error(f"Invalid number of iterations: {e}")
                return 

            # Ensure there are CSV files to process
            if not self.csv_files:
                messagebox.showerror("No CSV Files", "No CSV files found in the selected folder. Please ensure the folder contains CSV files.")
                logging.error("No CSV files found to process.")
                return

            # Perform Monte Carlo simulation
            arrays, names, trans_acc, trans_dec = self.monte_carlo.Arrays_division(self.csv_files)
            positions = self.monte_carlo.rnd_positions(arrays, n)
            self.monte_carlo.cut_and_save(arrays, positions, n, self.test_folder, self.train_folder, trans_acc, trans_dec)
            
            # Inform the user that the process has completed successfully
            messagebox.showinfo("Process Complete", f"Database creation completed successfully with {n} iterations.")
            logging.info(f"Database creation completed successfully with {n} iterations.")

        except Exception as e:
            # Log the error and display an error message to the user
            logging.error(f"Error in database creation: {e}")
            messagebox.showerror("Database Creation Error", f"An error occurred while creating the database: {e}")
    
    
    def clear_database(self):
        
        """
        Clears the content of the Test and Train folders by removing all files and directories.
        Provides user feedback on the success or failure of the operation.
        """
        
        try:
            # Clear the Test folder
            if os.path.exists(self.test_folder):
                for file_name in os.listdir(self.test_folder):
                    file_path = os.path.join(self.test_folder, file_name)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)  # Remove individual files
                            logging.info(f"Removed file: {file_path}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Remove directories
                            logging.info(f"Removed directory: {file_path}")
                    except Exception as e:
                        logging.error(f"Error removing {file_path}: {e}")
                        messagebox.showerror("File Removal Error", f"Error removing {file_path}: {e}")

            else:
                logging.warning(f"Test folder does not exist: {self.test_folder}")
                messagebox.showwarning("Folder Missing", f"The Test folder does not exist at {self.test_folder}.")
        
            # Clear the Train folder
            if os.path.exists(self.train_folder):
                for file_name in os.listdir(self.train_folder):
                    file_path = os.path.join(self.train_folder, file_name)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)  # Remove individual files
                            logging.info(f"Removed file: {file_path}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Remove directories
                            logging.info(f"Removed directory: {file_path}")
                    except Exception as e:
                        logging.error(f"Error removing {file_path}: {e}")
                        messagebox.showerror("File Removal Error", f"Error removing {file_path}: {e}")

            else:
                logging.warning(f"Train folder does not exist: {self.train_folder}")
                messagebox.showwarning("Folder Missing", f"The Train folder does not exist at {self.train_folder}.")
            
            # Inform the user that the operation was successful
            messagebox.showinfo("Database Cleared", "Test and Train folders have been cleared successfully.")
            logging.info("Test and Train folders have been cleared successfully.")

        except Exception as e:
            logging.error(f"Error in clearing the database: {e}")
            messagebox.showerror("Clear Database Error", f"An error occurred while clearing the database: {e}")
        
    
    
    
if __name__ == "__main__":
    root = tk.Tk();
    app = MonteCarloApp(root);
    root.mainloop();