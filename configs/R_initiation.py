import numpy as np
from typing import Literal


def _generate_vectors(theta, phi, *, cos_sign=None, include_base=False):
    """Construct vectors from polar angle(s) theta and directional matrix phi."""

    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    num_vectors, spatial_dim = phi.shape

    theta_values = np.asarray(theta, dtype=float)
    if theta_values.ndim == 0:
        theta_values = np.full(num_vectors, theta_values, dtype=float)
    elif theta_values.shape[0] != num_vectors:
        raise ValueError("theta must be a scalar or have one entry per phi row")

    cos_components = np.cos(theta_values)
    sin_components = np.sin(theta_values)

    if cos_sign is None:
        cos_sign = np.ones_like(cos_components)
    else:
        cos_sign = np.asarray(cos_sign, dtype=float)
        if cos_sign.shape != cos_components.shape:
            raise ValueError("cos_sign must match the number of vectors")

    vectors = np.column_stack((cos_sign * cos_components, sin_components[:, None] * phi))

    if include_base:
        base_vector = np.zeros(spatial_dim + 1, dtype=float)
        base_vector[0] = 1.0
        vectors = np.vstack((base_vector, vectors))

    return [vectors[i] for i in range(vectors.shape[0])]


def _azimuthal_directions(angles, *, extra_zeros=1):
    angles = np.asarray(angles, dtype=float)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    if extra_zeros > 0:
        directions = np.column_stack((directions, np.zeros((angles.size, extra_zeros), dtype=float)))
    return directions


_TETRAHEDRAL_DIRECTIONS = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
], dtype=float) / np.sqrt(3)


#########################################################
# Symmetric classes in R^3 and R^4
#########################################################
def _symmetric_classes_r4():
    # PRODUCING 5 vectors in R^4 in a fully rotationaly symmetric way
    theta = np.arccos(-1/4)                     # symmetric angle for 5 points on S³
    v1, v2, v3, v4, v5 = _generate_vectors(theta, _TETRAHEDRAL_DIRECTIONS, include_base=True)
    R_00 = _compute_R_00(v1, v2, v3, v4, v5)
    return R_00

def _symmetric_classes():
    theta = np.arccos(-1/3)       # polar angle
    phi = _azimuthal_directions([0, 2*np.pi/3, 4*np.pi/3])  # azimuth directions in R^3

    v1, v2, v3, v4 = _generate_vectors(theta, phi, include_base=True)
    R_00 = _compute_R_00(v1, v2, v3, v4)
    return R_00

#########################################################
# two cluster of close classes in R^3 and R^4
#########################################################
def _two_classes_close():
    # two vs two cluster of close classes in R^3
    theta = 7*np.pi/8       # polar angle
    phi = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
    ], dtype=float)
    cos_sign = np.array([1, 1, -1, -1], dtype=float)

    v1, v2, v3, v4 = _generate_vectors(theta, phi, cos_sign=cos_sign)

    R_00 = _compute_R_00(v1, v2, v3, v4)
    return R_00

def _two_vs_three_classes():
    theta_three = np.arccos(-0.3)               # open angle between clusters
    eps = 0.20                                  # closeness of the two-cluster
    beta_two = np.pi/3
    beta_three = np.pi/3
    phi_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    phi = np.array([
        [np.cos(beta_two), np.sin(beta_two), 0, 0],
        [np.cos(beta_three), np.sin(beta_three)*np.cos(phi_angles[0]), np.sin(beta_three)*np.sin(phi_angles[0]), 0],
        [np.cos(beta_three), np.sin(beta_three)*np.cos(phi_angles[1]), np.sin(beta_three)*np.sin(phi_angles[1]), 0],
        [np.cos(beta_three), np.sin(beta_three)*np.cos(phi_angles[2]), np.sin(beta_three)*np.sin(phi_angles[2]), 0],
    ], dtype=float)
    theta_values = np.array([eps, theta_three, theta_three, theta_three], dtype=float)

    v1, v2, v3, v4, v5 = _generate_vectors(theta_values, phi, include_base=True)

    R_00 = _compute_R_00(v1, v2, v3, v4, v5)
    return R_00

#########################################################
# One isolated class and rest symmetrically close classes in R^3 and R^4
#########################################################
def _three_classes_close(r):
    theta = np.arccos(-3/4)       # polar angle
    phi = _azimuthal_directions([0, 2*np.pi/3, -2*np.pi/3])  # azimuth directions in R^3

    v1, v2, v3, v4 = _generate_vectors(theta, phi, include_base=True)
    v1 = v1 * r # the isolated class is r times further away from the other three classes
    R_00 = _compute_R_00(v1, v2, v3, v4)
    return R_00

def _four_vs_one_classes(r):
    # PRODUCING 5 vectors in R^4, such that the last four are close to each other and the first is far away from the other four
    theta = 3*np.pi/4                           # tilt for the four-cluster
    v1, v2, v3, v4, v5 = _generate_vectors(theta, _TETRAHEDRAL_DIRECTIONS, include_base=True)
    v1 = v1 * r # the isolated class is r times further away from the other four classes
    R_00 = _compute_R_00(v1, v2, v3, v4, v5)
    return R_00

def _two_vs_two_vs_one():
    # returns 5 vectors in R^4, containing two clusters of two points and one isolated point

    # One isolated point (base vector) and two tight two-point clusters positioned apart
    delta = np.pi / 12                     # small angular separation within each cluster
    theta_close = np.pi / 3                # cluster close to the base class
    theta_far = 2 * np.pi / 3              # cluster further from the base class

    phi = np.array([
        [1.0, 0.0, 0.0],
        [np.cos(delta), np.sin(delta), 0.0],
        [0.0, 1.0, 0.0],
        [0.0, np.cos(delta), np.sin(delta)],
    ], dtype=float)

    theta_values = np.array([theta_close, theta_close, theta_far, theta_far], dtype=float)

    v1, v2, v3, v4, v5 = _generate_vectors(theta_values, phi, include_base=True)
    #v4 = v4 * 0.5


    R_00 = _compute_R_00(v1, v2, v3, v4, v5)
    return R_00
def _compute_R_00(v1, v2, v3, v4, v5=None):
    # --- Compute theta1, theta2, theta3 and Theta0,R_00=Theta0.T @ Theta0 ---
   # breakpoint()

    theta1, theta2, theta3 = v2 - v1, v3 - v1, v4 - v1
    theta4 = v5 - v1 if v5 is not None else None
    # Theta0 = [theta1, theta2, theta3]
    Theta0 = np.column_stack((theta1, theta2, theta3))  if v5 is None else np.column_stack((theta1, theta2, theta3, theta4))
    V = np.column_stack((v1, v2, v3, v4))  if v5 is None else np.column_stack((v1, v2, v3, v4, v5))

    return Theta0.T @ Theta0





def get_R_00(k, type:Literal['symmetric', 'two_classes_close', 'three_classes_close', 'two_vs_two_vs_one'] = 'symmetric'):
    if type == 'symmetric':
        if k == 3:
            return _symmetric_classes()
        elif k == 4:
            return _symmetric_classes_r4()

    elif type == 'two_classes_close':
        if k == 3:  
            return _three_classes_close(r = 1)
        elif k == 4:
            return _four_vs_one_classes(r = 1)

    elif type == 'three_classes_close':
        if k == 3:
            return _two_classes_close()
        elif k == 4:
            return _four_vs_one_classes(r = 3)
            #return _two_vs_three_classes()

    elif type == 'two_vs_two_vs_one':
        if k == 4:
            return _two_vs_two_vs_one()

    else:
        raise ValueError(f"Invalid R_00 type: {type}; choices are 'symmetric', 'two_classes_close', or 'three_classes_close' two_vs_two_vs_one")