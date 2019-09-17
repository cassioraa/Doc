import numpy as np
from numba import jit
from objective import *


def compute_ss_dist(transition_matrix): #, initial_dist=None):

	'''Descrição: calcula a distribuição estacionária da cadeia de markov com matriz de transição pi

	Phi = Phi transition_matrix

	Argumentos:
	------------

	Retorna:
	----------
	'''

	n = len(transition_matrix)#np.shape(transition_matrix)[0]
	
	# if initial_dist is None:
	Pi = np.ones((1,n))/n
	# else:
	# 	Pi = initial_dist

	norma, tol, it = 1, 1e-2, 0

	while norma>tol:

		Pi_ss = np.dot(Pi, transition_matrix)

		norma = max(abs(Pi_ss[0,:] - Pi[0,:]))
		# print(norma)
		Pi = Pi_ss.copy()

	return Pi_ss

def compute_L(y_grid, pi):

	'''Descrição: Computa o equilíbrio no mercado de trabalho

	Argumentos:
	------------

	Retorna:
	------------
	'''

	Pi_ss = compute_ss_dist(pi)

	L = 0
	for i_y, y in enumerate(y_grid):

		L += y*Pi_ss[0,i_y]

	return L

@jit(nopython=True)
def compute_k(r, alpha, delta):

	'''Descrição: computa o capital por trabalhador, baseado na otimização da firma '''
	k = ((r+delta)/alpha)**(1/(alpha-1))
	return k

# @jit(nopython=True)
def compute_K(r, alpha, delta, L):
	'''Descrição: computa o capital, baseado na otimização da firma '''

	k = (((r+delta)/alpha)**(1/(alpha-1)))
	K = k*L
	return K

@jit(nopython=True)
def compute_w(k,alpha):


	'''Descrição: Computa o salário w da condição de otimização da firma


	Argumentos:
	-------------
	K: capital (float)
	L: trabalho
	alpha: participação do trabalho (float)

	Retorna:
	-------------
	w: taxa de salário (float)'''
	w = (1-alpha) * k**(alpha)
	return w

@jit(nopython=True)
def compute_Tv_Tg(Fobj, a_grid, y_grid):

	'''Descrição:'''
	n_y = len(y_grid)
	n_a = len(a_grid)

	Tv = np.zeros((n_y, n_a))
	Tg = np.zeros((n_y, n_a))
	for i_y in range(n_y):
		for i_a in range(n_a):
			Tv[i_y, i_a] = np.max(Fobj[i_y, i_a,:])
			Tg[i_y, i_a] = np.argmax(Fobj[i_y, i_a,:])
	return Tv, Tg

def compute_v_g(a_grid, y_grid, w, r, beta, sigma, pi, v):

	norma, tol = 1, 1e-2

	while norma>tol:

		# calcule a função objetivo
		Fobj = build_objective(a_grid, y_grid, w, r, beta, sigma, pi, v)

		# Computa a equação de Bellman
		Tv, Tg = compute_Tv_Tg(Fobj, a_grid, y_grid)

		# Calcula a norma
		norma = np.max(abs(Tv- v))
		v = Tv.copy()

	return Tv, Tg

@jit(nopython=True)
def compute_Q(policy_function, pi):

	'''Descrição:

	Argumentos:
	------------
	policy_function:
	pi:

	Retorna:
	------------
	Q: '''

	n_y, n_a = len(policy_function[:,0]), len(policy_function[0,:])
	# n_y, n_a = np.shape(policy_function)

	Q = np.zeros((n_a*n_y, n_a*n_y))

	for i_y in range(n_y):
		for i_a in range(n_a):
			current_state = i_y*n_a + i_a
			for i_yy in range(n_y):
				for i_aa in range(n_a):
					next_state = i_yy*n_a + i_aa
					if i_aa == policy_function[i_y,i_a]:
						Q[current_state, next_state] += pi[i_y,i_yy]
	return Q

@jit(nopython=True)
def compute_Ea(policy_function, a_grid, Phi_ss):

	'''Description: compute the expectation of a
	
	\\sum_{i=0}^n_a a'[i] Phi[i]

	'''

	n_y, n_a = len(policy_function[:,0]), len(policy_function[0,:])
	# n_y, n_a = np.shape(policy_function)

	Ea = 0
	for i_y in range(n_y):
		for i_a in range(n_a):
			
			state_y_a = i_y*n_a + i_a

			position = int(policy_function[i_y, i_a])

			Ea += a_grid[position]*Phi_ss[0,state_y_a]

	return Ea

def compute_d(y_grid, a_grid, policy_function, r, alpha, delta, pi, Phi_ss):

	L = compute_L(y_grid, pi)
	K = compute_K(r, alpha, delta, L)
	Ea= compute_Ea(policy_function, a_grid, Phi_ss)
	d = K - Ea

	return d