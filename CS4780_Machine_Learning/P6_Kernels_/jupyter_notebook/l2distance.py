import numpy as np
from numpy.matlib import repmat
import torch

def l2distance(X, Z=None):
    x_copy = X.detach().numpy()

    z_copy = None
    if Z is not None:
        z_copy = Z.detach().numpy()

    dist_mat = l2distance_numpy(x_copy, z_copy)

    dist_mat_torch = torch.tensor(dist_mat).float()

    return dist_mat_torch

def l2distance_numpy(X, Z=None):
    """
    function D=l2distance(X,Z)

    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: dxn data matrix with n vectors (columns) of dimensionality d
    Z: dxm data matrix with m vectors (columns) of dimensionality d

    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)

    call with only one input:
    l2distance(X)=l2distance(X,X)
    """

    if Z is None:
        ## fill in code here
        ## << ms2666
        n, d = X.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
        D = D1 + repmat(s1.T, n, 1)
        np.fill_diagonal(D, 0)
        D = np.sqrt(np.maximum(D, 0))
        ## >> ms2666
    else:
        ## fill in code here
        ## << ms2666
        n, d = X.shape
        m, _ = Z.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
        D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
        D = D1 + repmat(s2, n, 1)
        D = np.sqrt(np.maximum(D, 0))
        ## >> ms2666
    return D