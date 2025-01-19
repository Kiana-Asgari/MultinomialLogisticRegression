import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import sqrtm
from multinomial_logistic.MLE_empirical.utils.data_generation import generate_data
import math
import seaborn as sns
import matplotlib as mpl

def scatter_plot_data(alpha, k, k_0, d, title):
    print(f"Generating data for scatter plot: alpha={alpha}, k={k}, k_0={k_0}, d={d}")
    zeros_pad = np.zeros((k, d-k))  # k x (d-k) matrix of zeros
    if title == "symmetric":
        beta_0 = np.array([0,-1])
        beta_1 = np.array([math.cos(math.pi/6), math.sin(math.pi/6)])
        beta_2 = np.array([-math.cos(math.pi/6), math.sin(math.pi/6)])
    elif title == "non-symmetric":
        beta_0 = np.array([0,1])
        beta_1 = np.array([math.cos(math.pi/6), math.sin(math.pi/6)])
        beta_2 = np.array([-math.cos(math.pi/6),math.sin(math.pi/6)])
    elif title == "two classes close":
        beta_0 = np.array([-1,0])
        beta_1 = np.array([math.cos(math.pi/12), math.sin(math.pi/12)])
        beta_2 = np.array([math.cos(math.pi/12), -math.sin(math.pi/12)])
    Theta_0 = np.array([beta_1 -beta_0, beta_2 -beta_0])
    norm_base= (Theta_0@Theta_0.T)[0,0]
    print((Theta_0@Theta_0.T)/norm_base)
    Theta_0 = np.hstack([Theta_0/norm_base, zeros_pad])  # concatenate horizontally to get k x d matrix
    print(Theta_0.shape)
    X, Y_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0)
    Y = np.argmax(Y_onehot, axis=1) + np.max(Y_onehot, axis=1)





    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7}) 
    mpl.rcParams.update({
        'text.usetex': True,            # For LaTeX rendering
        'font.family': 'serif',         # Use serif font family
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',       # Use Computer Modern math font
        'figure.dpi': 120,              
        'figure.figsize': (7, 5),       
        'axes.labelsize': 16,           
        'axes.titlesize': 16,           
        'xtick.labelsize': 14,          
        'ytick.labelsize': 14,          
        'legend.fontsize': 14,          
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'font.size': 14,                
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}', # Added bm package
        'mathtext.default': 'regular',   # Use regular (serif) font for math
        'axes.formatter.use_mathtext': True,  # Use mathtext for axis formatting
    })
    colors = ['darkred', 'darkblue', 'darkgreen', 'darkpurple']  # specify one color per class
    plt.scatter(X[:, 0],
                X[:, 1],
                s=10,
                c=[colors[int(y)] for y in Y])
    
    # Add arrows for beta vectors
    origin = np.array([0, 0])
    plt.arrow(origin[0], origin[1], beta_0[0], beta_0[1], 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.arrow(origin[0], origin[1], beta_1[0], beta_1[1], 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.arrow(origin[0], origin[1], beta_2[0], beta_2[1], 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Optional: Add labels for the arrows
    plt.text(beta_0[0], beta_0[1], r'$\beta_0$', fontsize=12)
    plt.text(beta_1[0], beta_1[1], r'$\beta_1$', fontsize=12)
    plt.text(beta_2[0], beta_2[1], r'$\beta_2$', fontsize=12)
    
    # Make sure the plot is centered and has equal aspect ratio
    plt.axis('equal')
    
    fig_dir = os.path.join(os.path.dirname(__file__), "figures", "data_visualization", title)
    # Create directory if it doesn't exist
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f"scatter_plot_alpha{alpha}_k{k}_k0{k_0}_d{d}.png"))
    print(f"Saved to {fig_dir}/scatter_plot_alpha{alpha}_k{k}_k0{k_0}_d{d}.png")