import numpy as np
import matplotlib.pyplot as plt
import time

from MyFunctions import *

#=====================
# Programa principal 
#=====================

# == Definindo alguns parâmetros == #

N = 80 # número de estados
w_grid = np.linspace(10, 20, N)

b = 6.0  # seguro desemprego
pi = 0.2 # probabilidade de ser demitido

sigma = 2.0
beta  = 0.98

start = time.time()

Tv = [np.zeros(N)]

Tv, g = bellman(w_grid, sigma, beta, pi, b, Tv)

end = time.time()

print('Tempo gasto: {:7.5f} segundos'.format(end-start))


# Figura

fig, ax1 = plt.subplots(figsize=(10,5))

ax1.set_ylabel('Função valor', color='blue')
ax1.set_xlabel('Salário', color='k')
ax1.tick_params(axis='y', direction='in', labelcolor='blue')
ax1.grid(linestyle='--')
ax1.plot(w_grid,Tv[-1], color='blue')
ax1.set_xlim(10,20)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(w_grid, g, color='red')
ax2.set_ylabel('Função política', color='red')
ax2.tick_params(direction='in', labelcolor='red')

plt.show()



#

M = np.zeros((N,N))

for i in range(N):
    
    for j in range(N):
        
        if g[i]==0:
            M[i,j] = f(w_grid[i], N)
        
        elif g[i] == 1 and j==i:
            M[i,j] = 1-pi
        
        elif g[i] == 1 and j==0:
            M[i,j] = pi
            
        else: 
            M[i,j] = 0


# Calcula distribuição Phi do estado estacionário

Phi = np.ones((1,N))/N

j = 0
norma = 1

while norma> 1e-6:
    
    Phi_new = Phi.dot(M)
    
    norma = max(abs(Phi_new[0,:] - Phi[0,:]))
    Phi = Phi_new
    
    print("iteração: {:3d} norma: {:8.5f} ".format(j,norma))
    
    j += 1
    
    
    
# Figura

fig, ax = plt.subplots()

ax.bar(list(w_grid),Phi[0,:], color='k')
ax.tick_params(direction='in')

plt.show()

# proporção de empregados em steady-state
e=0
for j in range(N):
    e = e + g[j] * Phi_new[:,j]
    
desemprego = 1 - e
print('desemprego', np.round(desemprego, decimals=2))