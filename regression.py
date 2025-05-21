import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from IQAPerformance import IQAPerformance
import warnings
import logging


def func1(x, beta0, beta1, beta2, beta3, beta4):
    return beta0*(0.5-1./(1+np.exp(beta1*(x-beta2))))+beta3*x+beta4


def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore")     # do not print warnings by genetic algorithm
    val = func1(Score, *parameterTuple)
    return np.sum((DMOS - val) ** 2.0)


def generate_Initial_Parameters(x, y):
    # min and max used for bounds
    beta = np.empty([5], dtype='float')
    beta[0] = np.abs(np.max(y) - np.min(y))
    beta[1] = 30/np.std(x)
    beta[2] = np.mean(x)
    beta[3] = np.mean(y)
    beta[4] = 1
    
    return beta


def regression(func1, Score, DMOS):
    # generate initial parameter values
    geneticParameters = generate_Initial_Parameters(Score, DMOS)
    
    # curve fit the test data
    fittedParameters, pcov = curve_fit(func1, Score, DMOS, p0=geneticParameters, maxfev=500000)

    logging.info('fittedParameters:{}'.format(fittedParameters))
    
    modelPredictions = func1(Score, *fittedParameters)
    
    return modelPredictions, fittedParameters

