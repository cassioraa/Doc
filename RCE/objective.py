import numpy as np 
from numba import jit
from util import *

@jit(nopython=True)
def build_objective(a_grid, y_grid, w, r, beta, sigma, pi, v):

	''' Descrição: Constrói a função objetivo
	$
	F(a,y| w, r, beta, sigma, pi) = U(w*y+(1+r)*a - a') + \\beta \\sum_{y' \\in Y } \\pi(y'|y) v(a',y'| \\Phi)
	$

	para cada estado da dotação y corrente;
	para cada estado do capital a corrente; e
	para cada estado do capital a' do próximo período.

	Argumentos:
	------------

	Retorna:
	------------
	F'''

	n_a = len(a_grid)
	n_y = len(y_grid)


	Fobj = np.zeros((n_y, n_a, n_a))

	for i_y, y in enumerate(y_grid):

		for i_a, a in enumerate(a_grid):
			for i_aa, aa in enumerate(a_grid):
				c = w*y + a - aa/(1+r)
				# Restrição de não negatividade
				if c<=0: 
					Fobj[i_y, i_a, i_aa] = -np.inf
				else:
					Fobj[i_y, i_a, i_aa] = util(c, sigma) + beta * np.dot(pi[i_y,:], v[:, i_aa])
	return Fobj