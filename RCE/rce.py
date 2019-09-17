import numpy as np 
import matplotlib.pyplot as plt
import time 

from objective import *
from compute import *

# Matriz de transição (de markov) dos estados (N=2) dos indíviduos
pi = np.array([
                [2/3, 1/3],
                [1/3, 2/3]
                ])

# Desconto intertemporal
beta  = 0.9
delta = 0.04
alpha = 0.3
sigma = 2

# Grid para dotação de trabalho
y_min, y_max, n_y = 1e-2, 1, len(pi)
y_grid = np.linspace(y_min, y_max, n_y)

# Grid para capital da família
a_min, a_max, n_a = 0, 5, 50
a_grid = np.linspace(a_min, a_max, n_a)

r_L, r_H = -delta, beta**(-1) - 1   # Taxa de juros inicial
value_function = np.zeros((n_y, n_a))           # função valor  inicial

###################################################################
#------------------------------------------------------------------
###################################################################
start = time.time()

norma_r, tol_r, iteration = 1, 1e-6, 0 

while norma_r>tol_r:
	#========================================================
	# Passo 1) fixa o r e resolve o problema da família
	#========================================================

	r = (r_L+r_H)/2 
	# Encontramos a demanda de capital
	k = compute_k(r, alpha, delta)

	# Encontramos a taxa de salário
	w = compute_w(k, alpha)

	# Dado w e r, resolva o problema da família:
	value_function, policy_function = compute_v_g(a_grid, y_grid, w, r, beta, sigma, pi, value_function) 

	#=========================================================
	# Passo 2) compute a distribuição estacionaria Phi
	#=========================================================

	# A função polítca 'a' e a matriz 'pi' induz uma função de transição de markov Q
	Q = compute_Q(policy_function, pi)

	# Encontramos a distribuição estacionária associada a função de transição de markov Q
	Phi_ss = compute_ss_dist(Q)

	#=========================================================
	# Passo 3) Compute o excesso de demanda por capital
	#=========================================================

	d = compute_d(y_grid, a_grid, policy_function, r, alpha, delta, pi, Phi_ss)

	#=========================================================
	# Passo 4) Atualizando o chute r
	#=========================================================

	if d==0:
		break

	elif d>0:
		r_L = np.copy(r)
	elif d<0:
		r_H = np.copy(r)

	norma_r = abs(r_H - r_L)#max(abs(r_H - r_L), abs(d))
	iteration += 1

	print('iteration {:3d}, norma_r = {:23.20f}'.format(iteration, norma_r))

# print some info

print("-"*100)
print("\n The interest rate is:", r)
print("\n The capital-labor rate is:", k)
print("\n The wage rate is:", w)
print("-"*100)

end = time.time()

print('Elapsed time: {:7.5f} seconds'.format(end-start))


# plot some graphics

fig, ax = plt.subplots(figsize=(10,3))
ax.bar(range(len(Phi_ss[0])), Phi_ss[0],  color='k')
ax.tick_params(direction='in')
ax.set_title("Distribution of individuals")
plt.savefig("Figuras/dist.pdf")

# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(a_grid, value_function[0,:], color='k', label="$y_0$")
# ax.plot(a_grid, value_function[1,:], c='blue', label="$y_1$")
# ax.tick_params(direction='in')
# ax.set_title("Value Function")
# ax.legend()
# plt.savefig("Figuras/value.pdf")

# #todo: reta de 45 graus.

g0 = [int(i) for i in policy_function[0,:]]

g1 = [int(i) for i in policy_function[1,:]]

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(a_grid, a_grid[g0], color='k', label="$y_0$")
ax.plot(a_grid, a_grid[g1], color='blue', label="$y_1$")
ax.tick_params(direction='in')
ax.plot([0,5], [0,5], "--", color='k' )
ax.set_title("Policy Function")
ax.legend()
plt.savefig("Figuras/policy.pdf")


# fig, ax = plt.subplots(figsize=(10,8))
# ax.hist(a_grid[g0], density=True, bins=10)

# fig, ax = plt.subplots(figsize=(10,8))
# ax.hist(a_grid[g1], density=True, bins=10)


# integrando na renda 

Phi_0 = Phi_ss[0,0:50]
Phi_1 = Phi_ss[0,50:100]

fig, ax = plt.subplots(figsize=(10,8))
ax.bar(range(len(Phi_0)), Phi_0+Phi_1, color='k')
ax.set_title("Distribution of wealth")
ax.tick_params(direction='in')
ax.legend()
plt.savefig("Figuras/Distribution_wealth.pdf")


plt.show()

