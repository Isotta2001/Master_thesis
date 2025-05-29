# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:08:27 2025

@author: isotta
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

class CSVEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Editor")
        self.root.geometry("500x400")
        self.root.configure(bg="#D3E4CD")

        # Title
        title_label = tk.Label(self.root, text="CSV Editor", font=("Arial", 16, "bold"), bg="#D3E4CD")
        title_label.pack(pady=10)

        # Button to upload the file
        self.load_button = tk.Button(self.root, text="Load CSV File", command=self.load_file, bg="#A7C957", font=("Arial", 12))
        self.load_button.pack(pady=10)

        # Button to select range
        self.range_button = tk.Button(self.root, text="Select Range", command=self.open_range_selector, bg="#A7C957", font=("Arial", 12), state=tk.DISABLED)
        self.range_button.pack(pady=10)

        # Apply button to apply range cut
        self.apply_button = tk.Button(self.root, text="Apply", command=self.apply_cut, bg="#FFB562", font=("Arial", 12), state=tk.DISABLED)
        self.apply_button.pack(pady=10)

        # Button to save file
        self.save_button = tk.Button(self.root, text="Save File", command=self.save_file, bg="#A7C957", font=("Arial", 12), state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Label for status
        self.status_label = tk.Label(self.root, text="Load a file to start", bg="#D3E4CD", font=("Arial", 10))
        self.status_label.pack(pady=10)

        # Initialize variables
        self.data = None
        self.filtered_data = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            # Load CSV file
            self.data = pd.read_csv(file_path)

            # Make sure the first column is a datetime object for proper filtering
            self.data.iloc[:, 0] = pd.to_datetime(self.data.iloc[:, 0])

            self.status_label.config(text=f"File loaded: {file_path.split('/')[-1]}")
            self.range_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)  # Disable save until filtered data is ready
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")

    def open_range_selector(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load a CSV file first!")
            return

        # Enable the Apply button after selecting the range
        self.apply_button.config(state=tk.NORMAL)

        # Create range selector window
        self.range_window = tk.Toplevel(self.root)
        self.range_window.title("Select Cut Range")
        self.range_window.geometry("400x600")
        self.range_window.configure(bg="#F2EDD7")

        # Create a frame for better layout control
        frame = tk.Frame(self.range_window, bg="#F2EDD7")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="Specify Start and End Range", font=("Arial", 12, "bold"), bg="#F2EDD7").pack(pady=10)

        # Start: Date and Time or Frame Option
        tk.Label(frame, text="Start Option:", bg="#F2EDD7").pack(pady=5)
        self.start_option = tk.StringVar(value="datetime")  # default to datetime
        tk.Radiobutton(frame, text="Start by Date/Time", variable=self.start_option, value="datetime", bg="#F2EDD7", command=self.update_start_input).pack(pady=5)
        tk.Radiobutton(frame, text="Start by Frame", variable=self.start_option, value="frame", bg="#F2EDD7", command=self.update_start_input).pack(pady=5)

        # Start: Date and Time Inputs
        self.start_date_label = tk.Label(frame, text="Start Date (YYYY-MM-DD):", bg="#F2EDD7")
        self.start_date_label.pack(pady=5)
        self.start_date_entry = tk.Entry(frame, font=("Arial", 10), width=20)
        self.start_date_entry.pack(pady=5)

        self.start_time_label = tk.Label(frame, text="Start Time (HH:MM:SS):", bg="#F2EDD7")
        self.start_time_label.pack(pady=5)
        self.start_time_entry = tk.Entry(frame, font=("Arial", 10), width=20)
        self.start_time_entry.pack(pady=5)

        # Start: Frame Inputs (Initially hidden)
        self.start_frame_label = tk.Label(frame, text="Start Frame (Row number):", bg="#F2EDD7")
        self.start_frame_label.pack(pady=5)
        self.start_frame_entry = tk.Entry(frame, font=("Arial", 10), width=20)
        self.start_frame_entry.pack(pady=5)
        self.start_frame_label.pack_forget()
        self.start_frame_entry.pack_forget()

        # End: Date and Time or Frame Option
        tk.Label(frame, text="End Option:", bg="#F2EDD7").pack(pady=5)
        self.end_option = tk.StringVar(value="datetime")  # default to datetime
        tk.Radiobutton(frame, text="End by Date/Time", variable=self.end_option, value="datetime", bg="#F2EDD7", command=self.update_end_input).pack(pady=5)
        tk.Radiobutton(frame, text="End by Frame", variable=self.end_option, value="frame", bg="#F2EDD7", command=self.update_end_input).pack(pady=5)

        # End: Date and Time Inputs
        self.end_date_label = tk.Label(frame, text="End Date (YYYY-MM-DD):", bg="#F2EDD7")
        self.end_date_label.pack(pady=5)
        self.end_date_entry = tk.Entry(frame, font=("Arial", 10), width=20)
        self.end_date_entry.pack(pady=5)

        self.end_time_label = tk.Label(frame, text="End Time (HH:MM:SS):", bg="#F2EDD7")
        self.end_time_label.pack(pady=5)
        self.end_time_entry = tk.Entry(frame, font=("Arial", 10), width=20)
        self.end_time_entry.pack(pady=5)

        # End: Frame Inputs (Initially hidden)
        self.end_frame_label = tk.Label(frame, text="total number of frame:", bg="#F2EDD7")
        self.end_frame_label.pack(pady=5)
        self.end_frame_entry = tk.Entry(frame, font=("Arial", 10), width=20)
        self.end_frame_entry.pack(pady=5)
        self.end_frame_label.pack_forget()
        self.end_frame_entry.pack_forget()

    def update_start_input(self):
        if self.start_option.get() == "datetime":
            self.start_frame_label.pack_forget()
            self.start_frame_entry.pack_forget()
            self.start_date_label.pack(pady=5)
            self.start_date_entry.pack(pady=5)
            self.start_time_label.pack(pady=5)
            self.start_time_entry.pack(pady=5)
        else:
            self.start_date_label.pack_forget()
            self.start_date_entry.pack_forget()
            self.start_time_label.pack_forget()
            self.start_time_entry.pack_forget()
            self.start_frame_label.pack(pady=5)
            self.start_frame_entry.pack(pady=5)

    def update_end_input(self):
        if self.end_option.get() == "datetime":
            self.end_frame_label.pack_forget()
            self.end_frame_entry.pack_forget()
            self.end_date_label.pack(pady=5)
            self.end_date_entry.pack(pady=5)
            self.end_time_label.pack(pady=5)
            self.end_time_entry.pack(pady=5)
        else:
            self.end_date_label.pack_forget()
            self.end_date_entry.pack_forget()
            self.end_time_label.pack_forget()
            self.end_time_entry.pack_forget()
            self.end_frame_label.pack(pady=5)
            self.end_frame_entry.pack(pady=5)

    def apply_cut(self):
        try:
            # Retrieve start and end datetime or frame
            if self.start_option.get() == "datetime":
                start_date = self.start_date_entry.get()
                start_time = self.start_time_entry.get()
                if not start_date or not start_time:
                    raise ValueError("Both start date and start time must be provided.")
                start_datetime = f"{start_date} {start_time}"
                start_datetime = pd.to_datetime(start_datetime)
                self.filtered_data = self.data[self.data.iloc[:, 0] >= start_datetime]
            else:
                start_frame = int(self.start_frame_entry.get())
                self.filtered_data = self.data.iloc[start_frame:]

            if self.end_option.get() == "datetime":
                end_date = self.end_date_entry.get()
                end_time = self.end_time_entry.get()
                if not end_date or not end_time:
                    raise ValueError("Both end date and end time must be provided.")
                end_datetime = f"{end_date} {end_time}"
                end_datetime = pd.to_datetime(end_datetime)
                self.filtered_data = self.filtered_data[self.filtered_data.iloc[:, 0] <= end_datetime]
            else:
                end_frame = int(self.end_frame_entry.get())
                self.filtered_data = self.filtered_data.iloc[:end_frame-1]

            # Update status and enable save
            self.status_label.config(text="Range applied successfully!")
            self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Error applying range: {e}")

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            # Save the filtered data
            self.filtered_data.to_csv(file_path, index=False)
            self.status_label.config(text=f"File saved: {file_path.split('/')[-1]}")

            # Reset application for next use
            self.data = None
            self.filtered_data = None
            self.load_button.config(state=tk.NORMAL)
            self.range_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.apply_button.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVEditorApp(root)
    root.mainloop()
