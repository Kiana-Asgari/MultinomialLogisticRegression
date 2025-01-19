from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline
from mnist_test.data_manipulation import get_cleaned_fashion_mnist_data, load_mnist_data
from sklearn.linear_model import LogisticRegression
import numpy as np

def fit_mle_vanilla_mnist(alpha, lambda_reg):
    pass

#winner for mnist: [3,5,8] fashion mnist: [2, 4, 6]
def find_R_00(classes_to_keep=[2, 4, 6], std_threshold=1e-4):
    #(x_train, y_train, y_train_one_hot), (x_test, y_test, y_test_one_hot) = get_cleaned_mnist_data(classes_to_keep, std_threshold)
    #(x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    (x_train, y_train, y_train_one_hot), (x_test, y_test, y_test_one_hot) = get_cleaned_fashion_mnist_data(classes_to_keep)


    y_train = np.argmax(y_train_one_hot, axis=1)+np.max(y_train_one_hot,axis=1)
    y_test = np.argmax(y_test_one_hot, axis=1)+np.max(y_test_one_hot,axis=1)


    logreg = LogisticRegression(
            multi_class='multinomial', 
            fit_intercept=False,
            solver='lbfgs',       # can also use 'sag' or 'saga' if data is large
            max_iter=500,
            verbose=1
    )

    logreg.fit(x_train, y_train)
    coefs_original     = logreg.coef_.copy()        # shape (3, d)
    
    # Get test accuracy (1 - error rate)
    test_accuracy = logreg.score(x_test, y_test)
    test_error = 1 - test_accuracy
    
    # Or alternatively, calculate error manually:
    # y_pred = logreg.predict(x_test)
    # test_error = np.mean(y_pred != y_test)
    
    print(f"Test error: {test_error:.4f}")
    
    base_coef = coefs_original[0]
    Theta_hat = coefs_original - base_coef
    print(Theta_hat @ Theta_hat.T)





    avg_Theta_hat, avg_norms, avg_test_error, avg_train_error = fit_mle_baseline(X_train_batch=x_train,
                                                                                  Y_train_batch=y_train_one_hot,
                                                                                  X_test_batch=x_test,
                                                                                  Y_test_batch=y_test_one_hot)
    print(avg_norms)
    print(avg_test_error)
    print(avg_train_error)
    R_00 = avg_Theta_hat @ avg_Theta_hat.T
    print('R_00:', R_00)
    return R_00
