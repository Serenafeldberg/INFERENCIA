import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


datos_hojas = "/Users/serena/Desktop/UDESA/Inferencia y estimacion/Teoria de la informacion/Dataset.xlsx"
data = pd.read_excel(datos_hojas)

clase1 = data[data['Clase'] == 1]
clase2 = data[data['Clase'] == 2]

largo1 = clase1['Largo'].tolist()
ancho1 = clase1['Ancho'].tolist()
largo2 = clase2['Largo'].tolist()
ancho2 = clase2['Ancho'].tolist()

def plot2D_gauss (x, y, muestras):
    # Crea una figura y un eje
    #FIJARSE EN EL CHAT
    pass


def normal_bidimensional ():
    media = np.array([2,2])
    cov = np.array([[1,1],[1,2]])

    muestra = np.random.multivariate_normal(media, cov, 1000)

    autovectores = np.linalg.eig(cov)[1]
    autovalores = np.linalg.eig(cov)[0]
    inv_autovalores = np.linalg.inv(np.diag(autovalores))

    x_new = []

    for x in muestra:
        x_new.append(np.dot(inv_autovalores,np.dot(autovectores.T, x)))

    x_new = np.array(x_new)

    fig, axes = plt.subplots(1,2, figsize=(10,5))

    axes[0].scatter(muestra[:,0], muestra[:,1], alpha=0.5)
    axes[0].set_title("Muestra original")

    axes[1].scatter(x_new[:,0], x_new[:,1], alpha=0.5)
    axes[1].set_title("Muestra transformada")

    for i in range(2):
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")

    plt.show()


normal_bidimensional()