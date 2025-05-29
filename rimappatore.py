
"""
Created on Thu Jan 23 10:50:29 2025

@author: isotta
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

class AxisRemapperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Axis Remapper")
        self.root.geometry("500x350")
        self.root.configure(bg="#D3E4CD")

        # Title
        title_label = tk.Label(self.root, text="Axis Remapping", font=("Arial", 16, "bold"), bg="#D3E4CD")
        title_label.pack(pady=10)

        # Button to upload the file
        self.load_button = tk.Button(self.root, text="Load File CSV", command=self.load_file, bg="#A7C957", font=("Arial", 12))
        self.load_button.pack(pady=10)

        # Button to save file
        self.save_button = tk.Button(self.root, text="Save remapped File", command=self.save_file, bg="#A7C957", font=("Arial", 12), state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Label for status
        self.status_label = tk.Label(self.root, text="Load a file to start", bg="#D3E4CD", font=("Arial", 10))
        self.status_label.pack(pady=10)

        # Initialize variables
        self.data = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("File CSV", "*.csv")])
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                sep = ',' if ',' in first_line else ';' if ';' in first_line else None
                if not sep:
                    raise ValueError("Separator not recognised. Use ',' or ';' in the CSV file.")

            self.data = pd.read_csv(file_path, delimiter=sep, header=None)
            self.status_label.config(text=f"File uploaded: {file_path.split('/')[-1]}")

            if self.data.shape[1] < 4:
                messagebox.showerror("Error", "The file should have at least 4 columns.")
                self.data = None
                return

            self.ask_column_order()
        except Exception as e:
            messagebox.showerror("Error", f"Error during file upload: {e}")

    def ask_column_order(self):
        self.order_window = tk.Toplevel(self.root)
        self.order_window.title("Column Order")
        self.order_window.geometry("400x350")
        self.order_window.configure(bg="#F2EDD7")

        tk.Label(self.order_window, text="What is column 2, 3, and 4? (Enter X, Y, Z in any order)", bg="#F2EDD7", font=("Arial", 12)).pack(pady=10)
        
        self.col_vars = []
        for i in range(3):
            var = tk.StringVar()
            entry = tk.Entry(self.order_window, textvariable=var, font=("Arial", 10), width=5)
            entry.pack(pady=5)
            self.col_vars.append(var)
        
        tk.Button(self.order_window, text="Next", command=self.ask_axis_mapping, bg="#A7C957", font=("Arial", 10)).pack(pady=10)

    def ask_axis_mapping(self):
        self.column_mapping = {self.col_vars[i].get().upper(): i + 1 for i in range(3)}
        self.order_window.destroy()
        
        self.mapping_window = tk.Toplevel(self.root)
        self.mapping_window.title("Axis Mapping")
        self.mapping_window.geometry("400x350")
        self.mapping_window.configure(bg="#F2EDD7")

        tk.Label(self.mapping_window, text="What do X, Y, Z correspond to? (AP, CS, ML)", bg="#F2EDD7", font=("Arial", 12)).pack(pady=10)
        
        self.axis_map = {}
        for axis in ["X", "Y", "Z"]:
            var = tk.StringVar()
            entry = tk.Entry(self.mapping_window, textvariable=var, font=("Arial", 10), width=5)
            entry.pack(pady=5)
            self.axis_map[axis] = var

        tk.Label(self.mapping_window, text="Select which axes to invert:", bg="#F2EDD7", font=("Arial", 12)).pack(pady=10)

        # Checkboxes to invert axes
        self.invert_flags = {}
        for axis in ["X", "Y", "Z"]:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.mapping_window, text=f"Invert {axis}", variable=var, bg="#F2EDD7")
            cb.pack(pady=5)
            self.invert_flags[axis] = var
        
        tk.Button(self.mapping_window, text="Confirm", command=self.remap_axes, bg="#A7C957", font=("Arial", 10)).pack(pady=10)

    def remap_axes(self):
        try:
            # Map the new column order based on user input
            mapping = {self.axis_map[axis].get().upper(): self.column_mapping[axis] for axis in self.axis_map}
            target_order = ["AP", "CS", "ML"]
            new_order = [mapping[key] for key in target_order]
            
            # First column stays in place, so keep it as is
            first_col = self.data.iloc[:, 0]
            
            # Reorder the columns based on the new order
            reordered_data = self.data.iloc[:, new_order]
            
            # Invert values if checkbox is selected
            for axis, invert in self.invert_flags.items():
                if invert.get():
                    column_index = self.column_mapping[axis] - 1  # Convert to 0-based index
                    reordered_data.iloc[:, column_index] *= -1
            
            # Concatenate the first column with the reordered and potentially inverted data
            self.data = pd.concat([first_col, reordered_data], axis=1)
            
            # Enable save button and update status
            self.save_button.config(state=tk.NORMAL)
            self.status_label.config(text="Columns remapped! You can save the file.")
            self.mapping_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error during mapping process: {e}")

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("File CSV", "*.csv")])
        if not file_path:
            return
        try:
            self.data.to_csv(file_path, index=False, header=False, sep=',')
            self.status_label.config(text=f"File saved: {file_path.split('/')[-1]}")
        except Exception as e:
            messagebox.showerror("Error", f"Error with saving the file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AxisRemapperApp(root)
    root.mainloop()
