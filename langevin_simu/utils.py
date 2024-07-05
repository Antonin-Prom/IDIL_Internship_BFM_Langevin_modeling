# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:16 2024

@author: CBS
"""
from scipy.optimize import differential_evolution
import numpy as np
def diff_evolution_fit(x, y, func, bounds=[(-1,1),(-1,1)], popsize=10, maxiter=1000, tol=0.01):
    ''' wrapper to use scipy.optimize.differential_evolution() 
        on data x,y, using function func 
            x,y    : data to fit 
            func   : ref. to function used to fit x,y
            bounds : limits for each parameter of func [(min,max), (min,max), ...]
            popsize: population size
        Ex: fit a noisy parabola:
        > def f(x,a,b):
            return a*x**2 + b
        > x = linspace(-5,5,100)
        > y = 3*x**2 + 5 + randn(len(x))
        > utils.diff_evolution_fit(x, y, f, bounds=[(-5,5),(-5,7)])
        Out: array([2.99152131, 4.95353557])
    '''
    def RMSE(params, *data):
        x,y = data
        return np.sqrt(np.mean((func(x, *params)-y)**2))

    diffevo = differential_evolution(RMSE, bounds, args=(x,y), popsize=popsize, maxiter=maxiter, tol=tol)
    popt = diffevo.x