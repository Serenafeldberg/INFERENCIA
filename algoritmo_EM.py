import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import random
from scipy.stats import multivariate_normal


def read_data():
    df = pd.read_excel('dataset.xlsx')

    c1 = {'Largo': [], 'Ancho': []}
    c2 = {'Largo': [], 'Ancho': []}

    for i in range(len(df)):
        if df['Clase'][i] == 1:
            c1['Largo'].append(df['Largo'][i]) 
            c1['Ancho'].append(df['Ancho'][i])
        else:
            c2['Largo'].append(df['Largo'][i]) 
            c2['Ancho'].append(df['Ancho'][i])
    return c1, c2

def calculate_varianze (c_1, c_2,):
    centroide1 = np.mean(c_1, axis = 0)
    centroide2 = np.mean(c_2, axis = 0)
    
    var1 = np.sum((c_1 - centroide1)**2) * (1/len(c_1))
    var2 = np.sum((c_2 - centroide2)**2) * (1/len(c_2))
    return var1 + var2

def update_clases(largos, anchos, centroides):
    c1 = []
    c2 = []
    for i in range(len(largos)):
        dist1 = np.linalg.norm([largos[i] - centroides[0][0], anchos[i] - centroides[0][1]])
        dist2 = np.linalg.norm([largos[i] - centroides[1][0], anchos[i] - centroides[1][1]])
        if dist1 < dist2:
            c1.append((largos[i], anchos[i]))
        else:
            c2.append((largos[i], anchos[i]))
    return c1, c2

def K_means(largos, anchos, max_iters=8):
    centroides = [(20, 20), (21, 21)]
    varianzas = []

    fig = plt.figure(figsize=(12, 8))
    
    for iteracion in tqdm(range(max_iters)):
        c1, c2 = update_clases(largos, anchos, centroides)
        ax = fig.add_subplot(3, 3, iteracion + 1)
        ax.set_title(f"Iteración {iteracion}")
        c1 = np.array(c1)
        c2 = np.array(c2)
        varianzas.append(calculate_varianze(c1, c2))
        ax.scatter(c1[:, 0], c1[:, 1], color='r', alpha=0.5)
        ax.scatter(c2[:, 0], c2[:, 1], color='b', alpha=0.5)
        ax.grid(True)
        ax.scatter(centroides[0][0], centroides[0][1], color='black', marker='x')
        ax.scatter(centroides[1][0], centroides[1][1], color='black', marker='x')
        centroide_nuevo_1 = (np.mean(c1[:, 0]), np.mean(c1[:, 1]))
        centroide_nuevo_2 = (np.mean(c2[:, 0]), np.mean(c2[:, 1]))
        centroides = [centroide_nuevo_1, centroide_nuevo_2]

    ax = fig.add_subplot(3, 3, 9)
    ax.set_title("Varianza")
    ax.plot(varianzas)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def E_step(largos, anchos, mu, sigma, pi, K):
    N = len(largos)
    responsibilities = np.zeros((N, K))

    for i in range(N):
        for j in range(K):
            responsibilities[i, j] = pi[j] * multivariate_normal.pdf([largos[i], anchos[i]], mu[j], sigma[j])

        if np.sum(responsibilities[i, :]) != 0:
            responsibilities[i, :] /= np.sum(responsibilities[i, :])
        else:
            responsibilities[i, :] = 1/K

    return responsibilities

def M_step(largos, anchos, mu, sigma, pi, K, responsibilities):
    new_mu = np.zeros((K, 2))
    new_sigma = np.zeros((K, 2, 2))
    new_pi = np.zeros(K)
    for k in range(K):
        new_mu[k] = np.sum(responsibilities[:, k][:, np.newaxis] * np.array([largos, anchos]).T, axis=0) / np.sum(responsibilities[:, k])
        new_sigma[k] = np.zeros((2, 2))
        for n in range(len(largos)):
            new_sigma[k] += responsibilities[n, k] * np.outer(np.array([largos[n], anchos[n]]) - new_mu[k], np.array([largos[n], anchos[n]]) - new_mu[k])
        new_sigma[k] /= np.sum(responsibilities[:, k])
        new_pi[k] = np.sum(responsibilities[:, k]) / len(largos)
    
    return new_mu, new_sigma, new_pi

def plot_data_and_normals(largos, anchos, mu, sigma, pi, i):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(largos, anchos, c='blue', alpha=0.5)
    
    x, y = np.meshgrid(np.linspace(np.min(largos), np.max(largos), 1000), np.linspace(np.min(anchos), np.max(anchos), 1000))
    pos = np.dstack((x, y))
    
    for k in range(len(mu)):
        rv = multivariate_normal(mu[k], sigma[k])
        plt.contour(x, y, rv.pdf(pos), levels= 10, colors=f'C{k}', label=f'Normal {k+1}')
    
    plt.xlabel('Largos')
    plt.ylabel('Anchos')
    plt.legend()
    plt.title(f'Iteracion, {i}' )
    plt.show()

def EM(largos, anchos):
    mu_0 = np.array([[20,30], [70,30]])
    sigma_0 = np.array([[[1, 0], [0, 0.5]], [[1, 0], [0, 0.5]]])  # definida positiva
    pi_0 = np.array([0.4, 0.6])  # suman 1
    N = len(largos)
    K = 2
    mus, sigmas, pis = [], [], []
    responsibilities = E_step(largos, anchos, mu_0, sigma_0, pi_0, K)
    plot_data_and_normals(largos, anchos, mu_0, sigma_0, pi_0, -1)

    for i in range(10):
        mu, sigma, pi = M_step(largos, anchos, mu_0, sigma_0, pi_0, K, responsibilities)
        mus.append(mu)
        sigmas.append(sigma)
        pis.append(pi)
        responsibilities = E_step(largos, anchos, mu, sigma, pi, K)
        plot_data_and_normals(largos, anchos, mu, sigma, pi, i)
    return None


def main():
    c1, c2 = read_data()
    largos = c1['Largo'] + c2['Largo']
    anchos = c1['Ancho'] +  c2['Ancho']
    # K_means(largos, anchos)
    EM(largos, anchos)

if __name__ == "__main__":
    main()