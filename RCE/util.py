import numpy as np
from numba import jit

# função utilidade
@jit(nopython=True)
def util(c, sigma):
	'''Descrição: Implementa função utilidade CRRA

	Parâmetros
	------------

	c: floating, consumo.
	sigma: coeficiente de aversão ao risco relativo

	retorna: floating, valor da utilidade u(c) \\in R '''
	if c>0:
		return (c**(1-sigma)-1)/(1-sigma)
	else:
		return -np.inf