import numpy as np
import matplotlib.pyplot as plt

u = np.random.uniform(0,1,1000)

log = []
for muestra in u:
    log.append(-np.log2(muestra))

plt.hist(log, bins=10)
plt.title('Histograma de frecuencias')
plt.show()

print(f'ESPERANZA DE LOG2: {np.mean(log)}')  # E(f(x))

print(-np.log2(np.mean(u)))  # f(E(x)) 

plt.hist(u, bins=10)
plt.title('Histograma de frecuencias')
plt.show()

print(f'ESPERANZA DE U: {np.mean(u)}')

# DESIGUALDAD DE JENSEN: f(E(x)) <= E(f(x))