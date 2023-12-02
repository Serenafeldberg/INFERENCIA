import numpy as np


def x_kalman (x, ti, tf, a):
    F = np.array([[1, ti], [0, 1]])
    G = np.array([0.5 * ti**2, ti])
    return np.dot(F, x) + np.dot(G, a)

def z_kalman (x, ek):
    H = np.array([1, 0])
    return np.dot(H, x) + ek

def prediction_step (F, G, x, P, sigma_a): 
    x = np.dot(F, x)
    P = np.dot(F, np.dot(P, F.T)) + np.dot(G, np.dot(sigma_a, G.T))
    return x, P

def medicion_step (H, x, P, z, sigma_e):
    y = z - np.dot(H, x)
    S = np.dot(H, np.dot(P, H.T)) + sigma_e
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
    x = x + np.dot(K, y)
    P = P - np.dot(K, np.dot (H, P))
    return x, P


P_inicion = np.array([[1, 0], [0, 1]])
x_inicial = np.array([0, 0])
sigma_a = np.array([[0.1, 0], [0, 0.1]])
sigma_e = 0.1


#MODEL COVARIANCE UPDATE


