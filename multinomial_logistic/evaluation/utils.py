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
    plt.hist(empirical_density, bins=100, density=True, label='Empirical Distribution', facecolor='none', edgecolor='blue')
    
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
               y_cap=None, irreducible_error=None, save_path=None, multiple_irreducible_error=False):
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
        if multiple_irreducible_error:
            plt.axhline(y=irreducible_error[i], color=color, linestyle='--', label='Irreducible Error')
        
        # Plot empirical data with same color but dotted line
        if empricial_data_batch is not None and i < len(empricial_data_batch):
            emp_data = np.asarray(empricial_data_batch[i]).flatten()
            plt.plot(x_data, emp_data, color=color, marker='o', linestyle='None', label=f"{legend} (empirical)")
    
    if irreducible_error is not None and not multiple_irreducible_error:
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
def custom_linspace(start, end, n_points, dense_factor=3):
    """
    Creates a non-uniform linspace with denser points at start and end.
    
    Args:
        start (float): Starting value
        end (float): Ending value
        n_points (int): Total number of points
        dense_factor (int): How many times denser the end regions should be
    
    Returns:
        np.array: Non-uniform spaced array
    """
    # Calculate the range and segment sizes
    total_range = end - start
    segment_size = total_range / 5  # Divide into 5 segments
    
    # Calculate points per segment
    base_points = n_points // (2*dense_factor + 3)  # Points in regular segments
    dense_points = base_points * dense_factor       # Points in dense segments
    
    # Create segments
    first_dense = np.linspace(start, start + segment_size, dense_points)
    middle1 = np.linspace(start + segment_size, start + 2*segment_size, base_points)[1:]
    middle2 = np.linspace(start + 2*segment_size, start + 3*segment_size, base_points)[1:]
    middle3 = np.linspace(start + 3*segment_size, start + 4*segment_size, base_points)[1:]
    last_dense = np.linspace(start + 4*segment_size, end, dense_points)
    
    # Combine all segments
    return np.concatenate([first_dense, middle1, middle2, middle3, last_dense])