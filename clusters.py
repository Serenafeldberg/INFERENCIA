import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datos_hojas = "/Users/serena/Desktop/UDESA/Inferencia y estimacion/Teoricas/Dataset.xlsx"
data = pd.read_excel(datos_hojas)

anchos = data['Ancho'].tolist()
largos = data['Largo'].tolist()

clase1 = data[data['Clase'] == 1]
clase2 = data[data['Clase'] == 2]

largo1 = clase1['Largo'].tolist()
ancho1 = clase1['Ancho'].tolist()
largo2 = clase2['Largo'].tolist()
ancho2 = clase2['Ancho'].tolist()


# Grafico de los datos
def grafico_datos (ancho, largo, c1 = None, c2 = None):
    plt.scatter (ancho, largo, color = 'black')
    if c1 != None and c2 != None:
        plt.scatter(c1[0], c1[1], color = 'red')
        plt.scatter(c2[0], c2[1], color = 'blue')
    plt.xlabel('Ancho')
    plt.ylabel('Largo')
    plt.title('Datos')
    plt.show()

def centroides (ancho, largo):
    c1 = (np.random.uniform(min(ancho), max(ancho)), np.random.uniform(min(largo), max(largo)))
    c2 = (np.random.uniform(min(ancho), max(ancho)), np.random.uniform(min(largo), max(largo)))

    return c1, c2

def distancia (x, y, c1, c2):
    #retornar la clase que tiene menor distancia
    d1 = np.sqrt((x - c1[0])**2 + (y - c1[1])**2)
    d2 = np.sqrt((x - c2[0])**2 + (y - c2[1])**2)
    if d1 < d2:
        return c1
    else:
        return c2
    
def varianza (ancho, largo, n):
    #retornar la varianza de cada clase
    var = 0
    for i in range (n):
        var += (ancho[i] - np.mean(ancho))**2 + (largo[i] - np.mean(largo))**2
    if n == 0:
        return 0
    return var/n

def k_means (anchos, largos, c1_i, c2_i):
    c1 = c1_i
    c2 = c2_i
    varc1 = []
    varc2 = []
    suma = []
    for i in range (10):
        c1_ancho = []
        c1_largo = []
        c2_ancho = []
        c2_largo = []
        for (x, y) in zip(anchos, largos):
            c = distancia(x, y, c1_i, c2_i)
            if c == c1_i:
                c1_ancho.append(x)
                c1_largo.append(y)
            else:
                c2_ancho.append(x)
                c2_largo.append(y)

        c1_i = (np.mean(c1_ancho), np.mean(c1_largo))
        c2_i = (np.mean(c2_ancho), np.mean(c2_largo))

        varc1.append(varianza(c1_ancho, c1_largo, len(c1_ancho)))
        varc2.append(varianza(c2_ancho, c2_largo, len(c2_ancho)))
        suma.append(varc1[i] + varc2[i])                          #La idea es minimizar la suma de las varianzas entonces mientras esta vaya disminuyendo todo ok.
                                                                  #no pasa nada si una baja y la otra sube. 

    plt.plot(varc1, color = 'red')
    plt.plot(varc2, color = 'blue')
    plt.plot(suma, color = 'black')
    plt.show()

    plt.scatter(c1_ancho, c1_largo, color = 'red')
    plt.scatter(c2_ancho, c2_largo, color = 'blue')
    plt.scatter(c1_i[0], c1_i[1], color = 'black')
    plt.scatter(c2_i[0], c2_i[1], color = 'black')
    plt.show()



c1_inicial, c2_inicial = centroides(anchos, largos)
# grafico_datos(anchos, largos, c1_inicial, c2_inicial)

k_means(anchos, largos, c1_inicial, c2_inicial)

