import numpy as np
import random
import matplotlib.pyplot as plt

def energia (s):
    energia = 0

    #agregar borde de ceros a s
    s = np.insert(s, 0, 0, axis=0)
    s = np.insert(s, len(s), 0, axis=0)
    s = np.insert(s, 0, 0, axis=1)
    s = np.insert(s, len(s[0]), 0, axis=1)

    for i in range (1, len(s)-1):
        for j in range (1, len(s[0])-1):
            energia += s[i][j] * (s[i-1][j] + s[i+1][j] + s[i][j-1] + s[i][j+1])
            
    return -1/2*energia

def cambio_aleatorio (s):
    i = random.randint(0, len(s)-1)
    j = random.randint(0, len(s[0])-1)

    s[i][j] *= (-1)

    return s

def cociente (s1, s2, T):
    return np.exp(-(energia(s1) - energia(s2))/T)

def aceptar_cociente (s1, s2, T):
    if energia(s2) < energia(s1):
        return True
    else:
        prob = cociente(s1, s2, T)
        nro_uniforme = np.random.uniform(0, 1, 1)
        if nro_uniforme < prob:
            return True
        else:
            return False
        
def magnetismo (s):
    return np.sum(s)

def plot (time, f):
    plt.plot(f)
    plt.show()
        
def simular_proceso (M):
    # Fijar seed
    np.random.seed(1)

    # Crear matriz de 1 y -1 random
    s = np.random.randint(2, size=(10,10))
    s = np.where(s==0, -1, s)
    old_s = np.copy(s)

    f_acum = 0
    f = []

    for i in range (M):
        f_acum += magnetismo(old_s)
        if i == 0:
            f.append(f_acum)
        else:
            f.append(f_acum/(i))

        new_s = cambio_aleatorio(s)
        if aceptar_cociente(old_s, new_s, 4):
            old_s = np.copy(new_s)


    plot(np.linspace(0, M, M), f)

    return old_s

        

# Fijar seed
# np.random.seed(1)

# # Crear matriz de 1 y -1 random
# s = np.random.randint(2, size=(10,10))
# s = np.where(s==0, -1, s)

# old_s = np.copy(s)
# new_s = cambio_aleatorio(s)

# print(energia(old_s))
# print(energia(new_s))

# print(cociente(old_s, new_s, 4))
# print(aceptar_cociente(old_s, new_s, 4))

print(simular_proceso(10000)) # Deberia converger a 10 * 10 = 100


