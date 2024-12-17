import matplotlib.pyplot as plt
import numpy as np
import os

def plot_array(x_data, y_data, title, x_label, y_label, name, save_path=None):
        # Convert inputs to numpy arrays and ensure they're 1D
    x_data = np.asarray(x_data).flatten()
    y_data = np.asarray(y_data).flatten()
    
    if len(x_data) != len(y_data):
        raise ValueError(f"x_data and y_data must have the same length. Got {len(x_data)} and {len(y_data)}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, '-')  
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    #plt.xlim(left=min(x_data), right=max(x_data))  # Add this line

    
    # Create data directory if it doesn't exist
    data_dir = save_path
    os.makedirs(data_dir, exist_ok=True)
    
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    base_path = os.path.join(data_dir, f'{safe_name}.png')
    
    # Check if file exists and add number if necessary
    final_path = base_path
    counter = 1
    while os.path.exists(final_path):
        # Split the path into name and extension
        name, ext = os.path.splitext(base_path)
        final_path = f"{name}_{counter}{ext}"
        counter += 1
    
    plt.savefig(final_path)
    plt.show()

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# plot_array(data, "Sample Plot", "X-axis", "Y-axis", "plot.png")
