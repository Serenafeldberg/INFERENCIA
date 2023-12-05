import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def H (x):
    return -x*np.log2(x)-(1-x)*np.log2(1-x)

# x = np.linspace(0.01,0.99,100)
# values = []
# for elem in x:
#     values.append(H(elem))

# plt.plot(x,values)
# plt.xlabel('Probabilidad (p)')
# plt.ylabel('Entropia H(p)')
# plt.title('Entropia de una variable binaria')
# plt.show()

#####

datos_hojas = "/Users/serena/Desktop/UDESA/Inferencia y estimacion/Teoricas/Dataset.xlsx"
data = pd.read_excel(datos_hojas)

clase1 = data[data['Clase'] == 1]
clase2 = data[data['Clase'] == 2]

largo1 = clase1['Largo'].tolist()
ancho1 = clase1['Ancho'].tolist()
largo2 = clase2['Largo'].tolist()
ancho2 = clase2['Ancho'].tolist()

def histogram (x, y, bins):
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    plt.imshow(hist_2d.T, origin='lower')
    plt.xlabel('Largo')
    plt.ylabel('Ancho')
    plt.title('Histograma de frecuencias')
    plt.show()

def cant_x_bin (x, y, bins):
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    cant = []
    for row in hist_2d:
        for col in row:
            cant.append(col)

    return cant

def joint_entropy (x, y, bins):
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hist_2d = hist_2d/np.sum(hist_2d) #Matriz de probabilidades

    ancho = (x_edges[1] - x_edges[0])*(y_edges[1] - y_edges[0]) #CAMBIE
    
    joint_entropy = 0
    for i in range(len(x_edges)-1):
        for j in range(len(y_edges)-1):
            if hist_2d[i,j] != 0:
                joint_entropy = joint_entropy - hist_2d[i,j]*np.log2(hist_2d[i,j]/ancho)
    return joint_entropy

# histogram(largo1, ancho1, 5)
# print(cant_x_bin(largo1, ancho1, 5))
# print(joint_entropy(largo1, ancho1, 5))

def conditonal_entropy (x, y, bins):
    #Y|X
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hist_2d = hist_2d/np.sum(hist_2d) #Matriz de probabilidades

    ancho = y_edges[1] - y_edges[0] #CAMBIE

    conditonal_entropy = 0
    for i in range (len(x_edges)-1):
        for j in range (len(y_edges)-1):
            if hist_2d[i,j] != 0:
                conditonal_entropy -= hist_2d[i,j]*np.log2((hist_2d[i,j]/hist_2d[i].sum())/ancho)

    return conditonal_entropy

# print(conditonal_entropy(ancho1, largo1, 5)) #H(Y|X)
# print(conditonal_entropy(largo1, ancho1, 5)) #H(X|Y)

# def h_x (x, y, bins): #H(X) marginal
#     hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
#     ancho = x_edges[1] - x_edges[0]
#     entropia = 0
#     for i in range (len(x_edges)-1):
#         if hist_2d[i].sum() != 0:
#             entropia -= (hist_2d[i].sum()/np.sum(hist_2d))*np.log2((hist_2d[i].sum()/np.sum(hist_2d))/ancho)

#     return entropia

def h_x(X, n_bins=5):
    x, edges = np.histogram(X, bins=n_bins)
    ancho = edges[1] - edges[0]
    p_x = (x/np.sum(x)) 
    h_x = 0
    for p in p_x:
    # - sumatoria p(x) log (p(x))
        if p != 0:
            h_x -= p * np.log2(p/ancho) 
    return h_x

# print(joint_entropy(largo1, ancho1, 5))
# print(conditonal_entropy(ancho1, largo1, 5) + h_x(ancho1, 5)) #H(Y|X) + H(X) = H(largo|ancho) + H(ancho)

#-> conditional = -sum, sum (p(x,y) log (p(x|y)))


#PARA ANCHO1 Y ANCHO2
def choose_sample (datos):
    """create a smaller sample from original data randomly"""
    copy = datos.copy()
    sample = []
    for i in range (200):
        index = np.random.randint(0, len(copy))
        sample.append(copy[index])
        copy.pop(index)
    return sample

sample1 = choose_sample(ancho1)
sample2 = choose_sample(ancho2)

# print(joint_entropy(sample1, sample2, 5)) = -sum, sum (p(x,y) log(p(x,y)))

# print(h_x(sample2, sample1, 5) + h_x(sample1, sample2, 5)) # -> como son independientes deberia dar igual. Como tengo pocas muestras me da una diferencia muy chica pero podria decir que son independientes

#PARA ANCHO1 Y LARGO1

# print(joint_entropy(ancho1, largo1, 5))
# print(h_x(largo1, ancho1, 5) + h_x(ancho1, largo1, 5)) # -> como la suma de las marginales es mayor a la conjunta, x e y son dependientes


# ENTROPIA RELATIVA DE KULLBACK-LEIBLER
# INFORMACION MUTUA

def I (x, y, bins):
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    hist_2d = hist_2d/np.sum(hist_2d) #Matriz de probabilidades

    info = 0
    for i in range (len(x_edges)-1):
        for j in range (len(y_edges)-1):
            if hist_2d[i,j] != 0:
                info += hist_2d[i,j]*np.log2(hist_2d[i,j]/(hist_2d[i].sum()*hist_2d[:,j].sum()))

    return info

# print(I(ancho1, largo1, 5)) #I(X,Y) = sum, sum (p(x,y) log (p(x,y)/p(x)p(y)))

# print(I(ancho1[:len(ancho2)], ancho2, 5)) #I(X,Y) -> como son independientes deberia dar 0. Como tengo pocas muestras me da una diferencia muy chica pero podria decir que son independientes

# I(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
# print(h_x(ancho1, 5) - conditonal_entropy(largo1, ancho1, 5)) #H(X) - H(X|Y)
# print(I(ancho1, largo1, 5)) #I(X,Y)

# I(X,X) = H(X) - H(X|X) = H(X)
# print(h_x(ancho1, largo1, 5), conditonal_entropy(ancho1, ancho1, 5)) #H(X)
# print(I(ancho1, ancho1, 5)) #I(X,Y)


# D(p||q) = sum,sum p(x,y) log (p(x,y)/q(x,y))


