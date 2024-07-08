import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox

# Connect to PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        dbname="fft_db",
        user="postgres",
        password="0Sms@800008",
        host="localhost",
        port="5432"
    )
    return conn

# Fetch data from the database
def fetch_data(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fft_data")
    data = cursor.fetchall()
    cursor.close()
    print(f"Fetched data: {data}")  # Logging fetched data
    return data

# Compute FFT
def compute_fft(data):
    data = np.array(data)
    fft_result = np.fft.fft(data)
    return fft_result

# Generate FFT and plot
def generate_fft(selected_indices):
    conn = connect_db()
    data = fetch_data(conn)
    conn.close()
    
    selected_data = [data[i][1:] for i in selected_indices]  # Exclude the ID column
    selected_data = np.array(selected_data).flatten()  # Flatten the data points
    
    print(f"Data for FFT: {selected_data}")  # Logging the data to be used for FFT
    fft_result = compute_fft(selected_data)
    
    print(f"FFT Result: {fft_result}")  # Logging FFT result
    
    plt.plot(np.abs(fft_result))
    plt.title("FFT Result")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()

# Create the GUI
def create_gui():
    root = Tk()
    root.title("FFT Generator")
    
    selected_indices = []

    def select_all():
        for var in check_vars:
            var.set(1)
        selected_indices.clear()
        selected_indices.extend(range(len(data)))
    
    def update_selection(index):
        if index in selected_indices:
            selected_indices.remove(index)
        else:
            selected_indices.append(index)
    
    # Fetch data
    conn = connect_db()
    data = fetch_data(conn)
    conn.close()
    
    # Create Select All button
    select_all_button = Button(root, text="Select All", command=select_all)
    select_all_button.pack(pady=5)
    
    # Create check buttons for each data point
    check_vars = []
    for index, row in enumerate(data):
        var = IntVar()
        check_button = Checkbutton(root, text=f"Data Point {index+1}: {row[1:]}", variable=var, command=lambda idx=index: update_selection(idx))
        check_button.pack(anchor='w')
        check_vars.append(var)
    
    # Create Generate FFT button
    generate_button = Button(root, text="Generate FFT", command=lambda: generate_fft(selected_indices))
    generate_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
