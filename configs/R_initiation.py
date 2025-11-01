import numpy as np
from typing import Literal


def _two_classes_close():
    theta = 7*np.pi/8       # polar angle
    phi = np.array([0, 7*np.pi/8, -7*np.pi/8 ])  # azimuths

    # Define vectors (regular tetrahedron on the unit sphere)
    v1 = np.array([np.cos(theta), np.sin(theta), 0,     0])
    v2 = np.array([np.cos(theta), np.sin(-theta), 0,    0])
    v3 = np.array([-np.cos(theta), 0, np.sin(theta),    0])
    v4 = np.array([-np.cos(theta), 0, np.sin(-theta),   0])
    R_00 = _compute_R_00(v1, v2, v3, v4)
    return R_00

def _symmetric_classes():
    theta = np.arccos(-1/3)       # polar angle
    phi = np.array([0, 2*np.pi/3, 4*np.pi/3])  # azimuths

    # --- Define four unit vectors for two classes close ---
    v1 = np.array([1, 0, 0,                                                                    0])
    v2 = np.array([np.cos(theta), np.sin(theta)*np.cos(phi[0]), np.sin(theta)*np.sin(phi[0]),  0])
    v3 = np.array([np.cos(theta), np.sin(theta)*np.cos(phi[1]), np.sin(theta)*np.sin(phi[1]),  0])
    v4 = np.array([np.cos(theta), np.sin(theta)*np.cos(phi[2]), np.sin(theta)*np.sin(phi[2]),  0])
    
    R_00 = _compute_R_00(v1, v2, v3, v4)
    return R_00

def _three_classes_close():
    theta = np.arccos(-3/4)       # polar angle
    phi = np.array([0, 2*np.pi/3, -2*np.pi/3])  # azimuths

    # --- Define four unit vectors for three classes close ---
    v1 = np.array([1, 0, 0,                                                                    0])
    v2 = np.array([np.cos(theta), np.sin(theta)*np.cos(phi[0]), np.sin(theta)*np.sin(phi[0]),  0])
    v3 = np.array([np.cos(theta), np.sin(theta)*np.cos(phi[1]), np.sin(theta)*np.sin(phi[1]),  0])
    v4 = np.array([np.cos(theta), np.sin(theta)*np.cos(phi[2]), np.sin(theta)*np.sin(phi[2]),  0])

    R_00 = _compute_R_00(v1, v2, v3, v4)
    return R_00

def _compute_R_00(v1, v2, v3, v4):
    # --- Compute theta1, theta2, theta3 and Theta0,R_00=Theta0.T @ Theta0 ---
    # theta_i \in R^d, i\in [k]
    # here k=3, d=4
    # Theta0 \in R^{d x k}
    # R_00 \in R^{k x k}
    theta1, theta2, theta3 = v2 - v1, v3 - v1, v4 - v1
    # Theta0 = [theta1, theta2, theta3]
    Theta0 = np.column_stack((theta1, theta2, theta3)) 
    R_00 = Theta0.T @ Theta0
    return R_00



def get_R_00(type:Literal['symmetric', 'two_classes_close', 'three_classes_close'] = 'symmetric'):
    if type == 'symmetric':
        return _symmetric_classes()
    elif type == 'two_classes_close':
        return _two_classes_close()
    elif type == 'three_classes_close':
        return _three_classes_close()
    else:
        raise ValueError(f"Invalid R_00 type: {type}; choices are 'symmetric', 'two_classes_close', or 'three_classes_close'")