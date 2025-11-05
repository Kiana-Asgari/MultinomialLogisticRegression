import numpy as np
import yaml
from pathlib import Path
from configs.R_initiation import get_R_00

def load_config(config_name, config_file='test_configs.yaml'):
    """
    Load a configuration from a YAML file.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration to load (e.g., 'K_1_example')
    config_file : str
        Name of the YAML file in the configs directory
        
    Returns:
    --------
    dict : Configuration dictionary with numpy arrays for matrix parameters
    """
    config_path = Path(__file__).parent / config_file
    
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if config_name not in all_configs:
        raise ValueError(f"Configuration '{config_name}' not found in {config_file}")
    
    params = all_configs[config_name]
    
    # Convert lists to numpy arrays for matrix parameters
    matrix_params = ['R_00', 'R_01_0', 'schur_0', 'S_0']
    for param in matrix_params:
        if param in params and params[param] is not None:
            params[param] = np.array(params[param])
    
    # Convert numeric parameters to correct types
    if 'tol' in params:
        params['tol'] = float(params['tol'])
    if 'lambda_reg' in params:
        params['lambda_reg'] = float(params['lambda_reg'])
    if 'alpha' in params:
        params['alpha'] = float(params['alpha'])
    
    return params


def example_configs(config_name, config_file='test_configs.yaml'):
    """
    Load and return state evolution parameters as a tuple.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration to load
    config_file : str
        Name of the YAML file in the configs directory
        
    Returns:
    --------
    tuple : (R_00, schur_0, R_01_0, lambda_reg, alpha, k, k_0, S_0, tol, max_iter, seed)
    """
    params = load_config(config_name, config_file)

    if params['k']>2 and params['R_00'] == 'symmetric':
        params['R_00'] = get_R_00(params['k'], 'symmetric')
        params['schur_0'] = params['R_00'].copy()
    
    return (
        params['R_00'],
        params['schur_0'],
        params['R_01_0'],
        params['lambda_reg'],
        params['alpha'],
        params['k'],
        params['k_0'],
        params.get('S_0', None),
        params['tol'],
        params['max_iter'],
        params['seed']
    )

