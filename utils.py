import numpy as np

def normalize_to(x, m, s):
    return (x-m)/s

def test_train_split(x,y,pct=.3):
    """
    pct: perecent to place in the test set
    """
    test_mask = np.array([False if np.random.rand() > pct else True for _ in range(len(x))])
    train_x, train_y = x[~test_mask], y[~test_mask]
    test_x, text_y = x[test_mask], y[test_mask]
    return train_x, train_y, test_x, text_y
