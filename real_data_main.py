import argparse
from tests.mnist_test import test_compute_features, test_fitting_whole_dataset
from tests.mnist_test import test_fitting_sampled_dataset, test_eval_esd_hessian 
from tests.mnist_test import test_eval_esd_hessian_theory, test_eval_esd_hessian_theory
from tests.mnist_test import log_esd_theory

#python real_data_main.py --alpha 20 --file-number 2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run MNIST tests with specified alpha value')
    parser.add_argument('--alpha', type=float, default=10,
                      help='Alpha value for ESD theory calculations (default: 10)')
    parser.add_argument('--file-number', type=int, default=1,
                      help='File number to process (default: 1)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    test_compute_features()
    test_fitting_whole_dataset()
    #test_fitting_sampled_dataset()
    #test_eval_esd_hessian()
    #test_eval_esd_hessian_theory()
    
    # Use the alpha value and file number from command line arguments
    alpha = args.alpha
    file_number = args.file_number
    log_esd_theory(file_number=file_number, alpha=alpha)


