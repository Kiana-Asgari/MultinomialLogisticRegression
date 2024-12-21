import matplotlib.pyplot as plt
import numpy as np
import os



def plot_array(x_data, y_data_batch, legends, title, x_label, y_label, name, irreducible_error=None, save_path=None, window_size=5):
    # Convert x_data to a numpy array and ensure it's 1D
    x_data = np.asarray(x_data).flatten()
    
    # Check if y_data_batch and legends have the same length
    if len(y_data_batch) != len(legends):
        raise ValueError(f"y_data_batch and legends must have the same length. Got {len(y_data_batch)} and {len(legends)}")

    plt.figure(figsize=(10, 6))
    
    # Plot each y_data with its corresponding legend
    for y_data, legend in zip(y_data_batch, legends):
        print(f"y_data: {y_data}")
        y_data = np.asarray(y_data).flatten()
        if len(x_data) != len(y_data):
            raise ValueError(f"x_data and each y_data must have the same length. Got {len(x_data)} and {len(y_data)}")
        
        # Smooth the y_data
        
        # Adjust x_data to match the length of smoothed_y_data
        smoothed_x_data = x_data[:len(y_data)]
        
        plt.plot(smoothed_x_data, y_data, label=legend)
    
    # Add a red dotted line at the irreducible error level
    if irreducible_error is not None:
        plt.axhline(y=irreducible_error, color='red', linestyle='--', label='Irreducible Error')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()  # Add legend to show labels for all lines
    
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
