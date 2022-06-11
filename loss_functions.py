import numpy as np


'''Collection of various loss functions'''

'''Assymetrical MSE function for XGBoost
Penalizes wrong prediction direction by alpha parameter'''
def assymetric_mse_train(y_true,y_pred):
    alpha = 100.

    sign = np.sign(np.divide(y_true,y_pred))
    residual = y_true - y_pred
    ratio = np.divide(y_pred, y_true)

    grad = np.where(sign<0,-2.0 * alpha * residual,-2.0*residual)
    # grad_2 = np.where(np.greater_equal(np.abs(ratio), 1.), (1/residual**2), residual)
    #
    # grad = grad_1 + grad_2

    hess = np.where(sign<0,2.0*alpha,2.0)
    # hess_2 = np.where(np.greater_equal(np.abs(ratio), 1.), (-2/residual**3), 1.0)
    #
    # hess = hess_1 + hess_2
    return grad, hess

'''Assymetrical MSE function for use in sklearn
Same as above, but returns average loss instead'''
def assymetric_mse_sklearn(y_true,y_pred):
    alpha = 100.
    loss = np.where(np.less(y_true * y_pred, 0),
                    alpha * y_pred**2 + np.square(y_true-y_pred),
                    np.square(y_true-y_pred)
                    )
    return np.average(loss, axis=-1)

'''Mse validation function, for use in LGBoost during validation'''
def assyemtric_mse_valid(y_true,y_pred):
    sign = np.sign(np.divide(y_true,y_pred)) #Needs to be near 0 when good pred
    residual = y_true - y_pred

    loss = np.where(sign<0, (residual**2)*10.0, residual**2)

    return "assymetric_mse_eval", np.mean(loss), False

'''Evaluation function
Returns % of samples that were guessed with the same sign'''
def eval_metric_signs(y_true,y_pred):

    y_true_sign = np.sign(y_true)
    y_pred_sign = np.sign(y_pred)

    metric = np.divide(np.abs(np.subtract(y_true_sign, y_pred_sign)), 2.)

    return np.multiply(
        np.divide(np.subtract(float(len(y_true)), np.sum(metric)),
                        float(len(y_true))),
        100.)