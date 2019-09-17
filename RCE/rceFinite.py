import numpy as np 
import matplotlib.pyplot as plt
import time 
import pandas as pd 

from compute import *

# Matriz de transição (de markov) dos estados (N=2) dos indíviduos
pi = np.array([
	            [2/3, 2/9, 1/9],
	            [1/6, 2/3, 1/6],
	            [1/9, 2/9, 2/3]
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
a_min, a_max, n_a = 0, 10, 5 
a_grid = np.linspace(a_min, a_max, n_a)

# Grid para o tempo
n_t = 3
t_grid = np.arange(n_t)	

r  = 0.1                         # Taxa de juros inicial
v0 = np.zeros((n_y, n_a, n_t+1)) # função valor  inicial

# dado v(t+1), t, w e r:
	# para cada estado corrente "y"
		# para cada estado corrente "a" 
			# para cada estado futuro "a'"
				# compute: u[wy+(1+r)a -a'] + beta* sum_y' pi v[a',y', t+1]
def objective(y_grid, a_grid, t_grid, v, t, w, r, sigma, beta, pi):

	n_y, n_a, n_t = len(y_grid), len(a_grid), len(t_grid)

	Fobj = np.zeros((n_y, n_a, n_a))

	for i_y, y in enumerate(y_grid):
		for i_a, a in enumerate(a_grid):
			for i_aa, aa in enumerate(a_grid):
				c = w*y + a - aa/(1+r)
				Fobj[i_y, i_a, i_aa] = util(c,sigma) + beta * np.dot(pi[i_y,:], v[:, i_aa, t+1])
	return Fobj


def compute_TvTg(Fobj, Tv, Tg, t):

	n_y, n_a, n_t = len(y_grid), len(a_grid), len(t_grid)

	for i_y in range(n_y):
		for i_a in range(n_a):
			Tv[i_y, i_a, t] = np.max(Fobj[i_y,i_a,:])
			Tg[i_y, i_a, t] = np.argmax(Fobj[i_y,i_a,:])

	return (Tv, Tg)

# main code

Tv = v0.copy()
Tg = v0.copy()

for i_t in range(1,n_t+1):
	t = n_t - i_t

	k = compute_k(r, alpha, delta)
	w = compute_w(k, alpha)

	Fobj   = objective(y_grid, a_grid, t_grid, Tv, t, w, r, sigma, beta, pi)
	Tv, Tg = compute_TvTg(Fobj, Tv, Tg, t)


print('\n\n')
print('\n\n')

print("*"*20)
print("*"*20)
print("Value Fuction")
print("*"*20)
print("*"*20)


for t in range(n_t):
	print("-"*60)
	print("i=",t)
	print("-"*60)
	print(pd.DataFrame(Tv[:,:,t]))


print('\n\n')
print('\n\n')

print("*"*20)
print("*"*20)
print("Policy Fuction")
print("*"*20)
print("*"*20)


for t in range(n_t):
	print("-"*60)
	print("i=",t)
	print("-"*60)
	print(pd.DataFrame(Tg[:,:,t]))

