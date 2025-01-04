import matplotlib.pyplot as plt
import numpy as np
import os


def plot_distribution(density, empirical_density, z_values, title, x_label, y_label, name, save_path=None):
    print("Shape of empirical_density:", np.array(empirical_density).shape)
    print("Shape of density_curve:", np.array(density).shape)
    print("Shape of z_real_values:", np.array(z_values).shape)
    plt.figure(figsize=(10, 6))
    
    # Plot theoretical density
    
    plt.plot(z_values, density, 'r-', label='Theoretical Density')
    
    # Plot histogram of empirical data
    plt.hist(empirical_density, bins=40, density=True, label='Empirical Distribution', facecolor='none', edgecolor='blue')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
        base_path = os.path.join(save_path, f'{safe_name}.png')
        
        final_path = base_path
        counter = 1
        while os.path.exists(final_path):
            name, ext = os.path.splitext(base_path)
            final_path = f"{name}_{counter}{ext}"
            counter += 1
        
        plt.savefig(final_path)
    #plt.show()



def plot_array(x_data, y_data_batch, empricial_data_batch, legends, title, x_label, y_label, name,\
               y_cap=None, irreducible_error=None, save_path=None, window_size=5):
    x_data = np.asarray(x_data).flatten()
    
    if y_data_batch.ndim == 1:
        y_data_batch = y_data_batch.reshape(1, -1)
    
    if len(y_data_batch) != len(legends):
        raise ValueError(f"y_data_batch and legends must have the same length. Got {len(y_data_batch)} and {len(legends)}")

    plt.figure(figsize=(10, 6))
    
    # Plot each pair of theoretical and empirical data
    for i, (y_data, legend) in enumerate(zip(y_data_batch, legends)):
        y_data = np.asarray(y_data).flatten()
        if len(x_data) != len(y_data):
            raise ValueError(f"x_data and each y_data must have the same length. Got {len(x_data)} and {len(y_data)}")
        
        # Get color from current plot
        line, = plt.plot(x_data, y_data, label=legend)
        color = line.get_color()
        
        # Plot empirical data with same color but dotted line
        if empricial_data_batch is not None and i < len(empricial_data_batch):
            emp_data = np.asarray(empricial_data_batch[i]).flatten()
            plt.plot(x_data, emp_data, color=color, marker='o', linestyle='None', label=f"{legend} (empirical)")
    
    if irreducible_error is not None:
        plt.axhline(y=irreducible_error, color='red', linestyle='--', label='Irreducible Error')
    if y_cap is not None:
        plt.ylim(top=y_cap)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    
    data_dir = save_path
    os.makedirs(data_dir, exist_ok=True)
    
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    base_path = os.path.join(data_dir, f'{safe_name}.png')
    
    final_path = base_path
    counter = 1
    while os.path.exists(final_path):
        name, ext = os.path.splitext(base_path)
        final_path = f"{name}_{counter}{ext}"
        counter += 1
    
    plt.savefig(final_path)
    #plt.show()

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# plot_array(data, "Sample Plot", "X-axis", "Y-axis", "plot.png")
