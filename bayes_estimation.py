import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import root_scalar
from scipy.stats import norm

datos_hojas = "/Users/serena/Desktop/UDESA/Inferencia y estimacion/Teoricas/Dataset.xlsx"
data = pd.read_excel(datos_hojas)

clase1 = data[data['Clase'] == 1]
clase2 = data[data['Clase'] == 2]

largo1 = clase1['Largo'].tolist()
ancho1 = clase1['Ancho'].tolist()
largo2 = clase2['Largo'].tolist()
ancho2 = clase2['Ancho'].tolist()


def clase1 ():

    x = np.linspace(0, 10,1000)
    def dist_exp(x, θ):
        return θ * np.exp(-θ * x)
    θ = 1
    plt.plot(x, dist_exp(x, θ))
    plt.title("Ploteando p(x|θ) con θ=1")
    plt.show()

    θ = np.linspace(0, 5, 1000)
    x = 2
    plt.title("Ploteando p(x=2|θ) con 0 ≤ θ ≤ 5")
    plt.plot(θ, dist_exp(x, θ))
    maximo = np.max(dist_exp(x, θ))
    #codigo que encuentra en que theta se encuentra el maximo
    max_theta = θ[np.where(dist_exp(x, θ) == maximo)]
    print(max_theta)
    plt.show()

    θ = 4
    x = np.random.exponential(θ, 500)
    suma = []
    for i in range(1,len(x)+1):
        suma.append(np.sum(x[:i])/i)
    plt.plot(suma)
    plt.plot(np.linspace(0,len(x),len(x)),[4]*len(x))
    plt.show()

    sigma_ac1 = np.std(ancho1)
    media_muestral = np.mean(ancho1)
    moo0 = np.mean(ancho1[:10])
    print(media_muestral)
    n = len(ancho1)
    def moo(sigma0,sigma,media_muestral,n,moo0):
        return ((n*(sigma0**2))/(n*(sigma0**2)+sigma**2))*media_muestral + ((sigma**2)/(n*(sigma0**2)+sigma**2))*moo0

    print(moo(1,sigma_ac1,media_muestral,n,moo0))

    def std(sigma0,sigma,n):
        return ((sigma0**2)*(sigma**2))/(n*(sigma0**2)+(sigma**2))

    print(std(1,sigma_ac1,n))


def random_dataset (dataset):
    dataset = random.shuffle(dataset)

    seventy = int(len(dataset)*0.7)
    
    return dataset[:seventy], dataset[seventy+1:]


def estimacion(x): 
    mu0 = np.mean(x[100:]) 
    sigma = np.std(x)
    sigma0 = 1 
    n = len(x) 
    _ = (n * (sigma0**2) + sigma**2)
    mu_n = (((n* (sigma0**2))/_) * np.mean(x)) + ((sigma**2 / _) * mu0) 
    return mu_n

print(estimacion(ancho1)) 

def univariable_case(sample1, sample2):
    # # agarro el 70% random
    # train1 = random.sample(sample1, int(len(sample1)*0.7))
    # set1 = set(sample1)
    # # agarro el otro 30%
    # sample1_ = random.shuffle(sample1)
    # test1 = [x for x in sample1 if x not in train1]
    # print("Test 1: ", len(test1))
    # print("Train 1: ", len(train1))

    # train2 = random.sample(sample2, int(len(sample2)*0.7))
    # test2 = [x for x in sample2 if x not in train2]

    random.shuffle(sample1)
    indice_division = int(0.7 * len(sample1))

    # Divide la lista en un 70% y un 30%
    train1 = sample1[:indice_division]
    test1 = sample1[indice_division:]


    random.shuffle(sample2)
    indice_division = int(0.7 * len(sample2))

    # Divide la lista en un 70% y un 30%
    train2 = sample2[:indice_division]
    test2 = sample2[indice_division:]


    media1 = np.mean(train1)
    media2 = np.mean(train2)
    sigma1 = np.std(train1)
    sigma2 = np.std(train2)

    # # busco N1 y N2 a partir de media1 y media2 y sigma1 y sigma2 
    pdf1 = lambda x: norm.pdf(x, loc=media1, scale=sigma1)
    pdf2 = lambda x: norm.pdf(x, loc=media2, scale=sigma2)

    # # Define la función que debe ser igualada a cero
    # intersection_function = lambda x: pdf1(x) - pdf2(x)

    def diff_pdf(x):
        return pdf1(x) - pdf2(x)

    # Encontrar la intersección de las dos PDF
    resultado = root_scalar(diff_pdf, bracket=[media1 - 3*sigma1, media2 + 3*sigma2])
    x_intersection = resultado.root


    # # Encuentra la intersección utilizando root_scalar
    # intersection = root_scalar(intersection_function, bracket=[media1 - 4*sigma1, media2 + 4*sigma2])

    # El resultado se encuentra en intersection.root
    # x_intersection = intersection.root
    cantidad_errores_test1 = 0
    cantidad_errores_test2 = 0
    if media1 < x_intersection:
        for i in test1:
            if i > x_intersection:
                cantidad_errores_test1 += 1
        for i in test2:
            if i < x_intersection:
                cantidad_errores_test2 += 1
    else:
        for i in test1:
            if i < x_intersection:
                cantidad_errores_test1 += 1
        for i in test2:
            if i > x_intersection:
                cantidad_errores_test2 += 1
    cantidad_tests = len(test1) + len(test2)
    error1 = cantidad_errores_test1 / len(test1)

    error2 = cantidad_errores_test2 / len(test2)

    error_tot = (error1*len(sample1) + error2*len(sample2)) / (len(sample1) + len(sample2))

    return error_tot

    """ 
    PERFORMANCE: 
    performance del sample total = performance(samlple1) *  cantidad de errores del sample 1/ cantidad de sample 1 + performance(sample2) * cantidad de errores del sample 2 / cantidad de sample 2
    performance de sample 1 = cantidad de sample 1 / cantidad de sample total  
    """

# univariable_case(ancho1, ancho2)

errores = []
for i in range(10):
    errores.append(univariable_case(ancho1, ancho2))
print("Error promedio: ", np.mean(errores))
print("Desvio estandar: ", np.std(errores))


# performance = X +- desvio_standar(todas las performances)/ raiz de n 
# n es la cantidad de muestras que tiene cada bloquecito 
# si tengo mi samples de largo 222, parto en bloquecitos de 22 y mi n = 10
# a cada bloquecito lo divido en train y test 

def estimador(sample1, sample2):
    if sample1 < sample2:
        sample2 = sample2[:len(sample1)]
    else:
        sample1 = sample1[:len(sample2)]

    bloques1 = []
    for i in range(0, len(sample1), 22):
        bloque = sample1[i:i+22]
        bloques1.append(bloque)
    
    bloques2 = []
    for i in range(0, len(sample2), 22):
        bloque = sample2[i:i+22]
        bloques2.append(bloque)

    performances = []
    # para cada bloque, lo divido en train y test 
    for i in range(len(bloques1)):
        sample1 = bloques1[i]
        sample2 = bloques2[i]
        print(univariable_case(sample1, sample2))
        # performances.append(1 - univariable_case(sample1, sample2))
    
    print("Performance: ", performances)
        
        
estimador(ancho1, ancho2)