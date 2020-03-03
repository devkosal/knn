import numpy as np
import utils
import pytest

x,y  = np.random.rand(100,4), np.random.randint(0,2,(100))


def test_normalize():
    mean = 0
    std = 1
    x_norm = utils.normalize_to(x,x.mean(),x.std())
    assert np.allclose(mean,x_norm.mean())
    assert np.allclose(std,x_norm.std())

def test_split():
    pct=.3
    x_train, y_train, x_test, y_test = utils.test_train_split(x,y,pct=pct)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert np.isclose(len(x_train)/len(x),pct)
