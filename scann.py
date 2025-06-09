import keras
import numpy as np

def remove_class(x_train, y_train, class_id, percentage):
    """
    This function removes a class fom a dataset.
    """
    to_delete=[]
    for i in range(0, len(y_train)):
        a=np.where(y_train[i]==1)
        if(a[0].item()==class_id and np.random.uniform()<percentage):
            to_delete.append(i)    
    to_delete=np.array(to_delete)
    x_train_new = np.delete(x_train, to_delete, axis=0)
    y_train_new = np.delete(y_train, to_delete, axis=0)
    
    return x_train_new, y_train_new

def sample_model(model, precision=0, dense_only = True):
    """
    This function modifies a model to sampled weights.
    """

    model_out = model
    for layer in model_out.layers:
        if dense_only and not isinstance(layer, keras.layers.Dense):
            continue
        weights=layer.get_weights()
        if not weights:
            continue
        #Ensure the weights have been clipped
        if(weights[0].max() > 1.0 or weights[0].min() < -1.0):
            continue
        weights_binary=weights
        if(precision==0):
            weight_compare = np.random.uniform(size=np.shape(weights[0]))
            a = weights[0] > weight_compare
            b = ((-1.0*weights[0]) > weight_compare)
        else:
            weight_compare = np.random.uniform(size=np.shape(weights[0]))
            weights_check = np.random.normal(weights[0], 1.0/precision, size=np.shape(weights[0]))
            a = weights_check > weight_compare
            b = ((-1.0*weights_check) > weight_compare)
        weights_binary[0] = a.astype(float) - b.astype(float)
        layer.set_weights(weights_binary)
    return model_out

def shannon_entropy(sample):
    """
    This function computes the Shannon entropy for the array sample.
    """
    n_sample = np.sum(sample)
    p_sample = sample/n_sample
    tol=1e-10
    tol_vec = np.where(p_sample==0, tol, 0)
    H=-np.sum(p_sample * np.log2(p_sample + tol_vec))
    return H
