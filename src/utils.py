import numpy as np

def one_hot_encode(labels, num_classes=10):
    """
    Convert a vector of labels into one-hot encoding.
    
    """
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot

def shuffle_data(X, Y):
    """
    Randomly shuffle X and Y while maintaining correspondence between samples.
     
    """
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], Y[idx]