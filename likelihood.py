import numpy as np
import matplotlib.pyplot as plt

def p (x, tita):  #Exp (tita)
    if x>= 0:
        return tita * np.exp(-tita*x)
    return 0

def plot (x, px):

    plt.plot(x, px)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Distribucion exponencial')
    plt.show()

#Plot P(x|tita) versurs x for tita = 1
x = np.linspace(0, 10, 100)
px = []
for elem in x:
    px.append(p(elem, 1))
#plot(x, px)

#Plot P(x|tita) versurs tita, (0 <= tita <= 5) for x = 2
tita = np.linspace(0, 5, 100)
px = []
for elem in tita:
    px.append(p(2, elem))

#plot(tita, px)

#Suppose that n samples x1, ..., xn are drawn independently from P(x|tita). Show that the maximum likelihood estimate of tita is given by:
# tita = n/sum(x1, ..., xn)

def simulation (n, tita):
    x = np.random.exponential(1/tita, n)

    tita_esperado = n / np.sum(x)
    return tita_esperado

def simulate ():
    n = [10, 100, 300, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 10000]
    n = np.arange(10, 10000, 10)
    titas = []
    tita = np.random.uniform(0, 5, 1)
    print(tita)
    for elem in n:
        titas.append(simulation(elem, tita))

    plt.plot(n, titas)
    plt.xlabel('n')
    plt.ylabel('tita')
    plt.title('Estimacion de tita')
    plt.show()

#simulate()

#EJERCICIO 2

def p2 (x, tita): #U(0, tita)
    if x >= 0 and x <= tita:
        return 1/tita
    return 0

def a ():
    """suppose that n samples D = {x1, ..., xn} are drawn independently from P(x|tita). 
    Show that the maximum likelihood estimate of tita is max(D)
    i.e., the value of maximum element in D.


    L (tita) = p(x1|tita)*p(x2|tita)*...*p(xn|tita)
    L (tita) = 1/tita^n

    -> derivo e igualo a cero: -n/tita^(n+1) = 0
    se cumple cuando tita->infinito --> tita = max(D)
    """

    def simulation2 (n, tita):
        x = np.random.uniform(0, tita, n)

        tita_esperado = np.max(x)
        return tita_esperado
    
    n = np.arange(10, 10000, 10)
    titas = []
    tita = np.random.uniform(0, 5, 1)
    print(tita)

    for elem in n:
        titas.append(simulation2(elem, tita))

    plt.plot(n, titas)
    plt.xlabel('n')
    plt.ylabel('tita')
    plt.title('Estimacion de tita')
    plt.show()

a()

def b ():
    """
    suppose that n = 5 points are drawn from the distributtion and the maximum value of which happens to be max(k) xk = 0.6.
    plot the likelihood p(D|tita) in the range 0 <= tita <= 1. Explain in word why you do not need to know the values of
    the other four points
    """

    



    

    

