import argparse
from tests.mnist_test import test_compute_features, test_fitting_whole_dataset
from tests.mnist_test import test_fitting_sampled_dataset, test_eval_esd_hessian 
from tests.mnist_test import test_eval_esd_hessian_theory, test_eval_esd_hessian_theory
from tests.mnist_test import log_esd_theory, log_error_theory, plots, log_error_empirical, plot_errors
from tests.mnist_test import log_bayesian_error, log_irreducible_error

#python real_data_main.py --alpha 20 --file-number 2
#python real_data_main.py --n-hidden 350 --feature-name tanh

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run MNIST tests with specified alpha value')
    parser.add_argument('--alpha', type=float, default=10,
                      help='Alpha value for ESD theory calculations (default: 10)')
    parser.add_argument('--file-number', type=int, default=1,
                      help='File number to process (default: 1)')
    parser.add_argument('--n-hidden', type=int, default=350,
                      help='Number of hidden units (default: 350)')
    parser.add_argument('--feature-name', type=str, default='tanh',
                      help='Feature name (default: tanh)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    #test_compute_features()
    #test_fitting_whole_dataset()
    #test_fitting_sampled_dataset()
    #test_eval_esd_hessian()
    #test_eval_esd_hessian_theory()

    #log_error_empirical(n_hidden=args.n_hidden, feature_name='relu')

    #alpha = args.alpha
    #file_number = args.file_number
    log_esd_theory(file_number=1, alpha=args.alpha)
    #log_error_theory(file_number=1, n_hidden=args.n_hidden, feature_name=args.feature_name)
    #plots(alpha=30.0)
    #plots(alpha=20.0)
    #plots(alpha=30.0)
    #log_error_empirical(n_hidden=args.n_hidden)
    #plot_errors(n_hidden=args.n_hidden)



