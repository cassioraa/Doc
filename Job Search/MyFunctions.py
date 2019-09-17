import numpy as np

# função utilidade
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
    
# densidade da uniforme discreta
def f(x, N = 1):
	return 1/N


def objective(w_grid, v, sigma, beta, pi, b):

	'''Descrição: Constrói função objetivo

	Parâmetros
	------------
	w_grid:
	v:
	sigma:
	beta:
	pi:
	'''
	N = len(w_grid)
	Fobj = np.zeros((N,2))


	Ev = 0
	for j in range(N):
	    Ev = Ev + v[j] * f(w_grid[j], N)

	for i in range(N):

	    # não aceita w
	    Fobj[i,0] = util(b,sigma) + beta * Ev

	    # aceita w
	    Fobj[i,1] = util(w_grid[i],sigma) + beta * ( (1-pi)*v[i] + pi * v[0])

	return Fobj

# atualiza v e g
def update_v_and_g(w_grid, sigma, beta, pi, b, vn):

	'''Descrição: Atualiza a função valor e a função política
	
	Parâmetros:
	-------------
	vn:

	retorna:'''
	N = len(w_grid)
	fobj = objective(w_grid, vn[-1], sigma, beta, pi, b)

	Tv = np.zeros(N)
	g  = np.zeros(N)
	for i in range(N):
	    
	    Tv[i] = max(fobj[i,:])
	    g[i]  = np.argmax(fobj[i,:])

	vn.append(Tv)

	return vn, g


# Operador de bellman
def bellman(w_grid, sigma, beta, pi, b, Tv):

	'''Descrição: Aplica o operador de bellman para encontrar o ponto fixo Tv = max({u(w) + Ev, u(b)+Ev})
	
	Parâmetros:
	-------------

	Tv
	Retorna:'''
	i = 0
	norma=1
	while norma>1e-6:
	    Tv, g = update_v_and_g(w_grid, sigma, beta, pi, b, Tv)
	    norma = max(abs(Tv[i+1] - Tv[i]))    
	    i += 1
	    print("iteração = {:3d} {:3} [norma] = {:3.2}".format(i,'...',norma))
	return (Tv, g)

