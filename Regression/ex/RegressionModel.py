#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 7, 2017

Super Class of different regression model

@author: Li Yulong 54927068
'''
import logging
import random
from warnings import catch_warnings

from cvxopt import matrix
from cvxopt import solvers
from numpy.dual import inv
from numpy.random.mtrand import uniform, normal
from scipy.linalg.misc import norm

import matplotlib.pyplot as plt
import numpy as np
from scipy.odr.models import quadratic


logging.basicConfig(level=logging.INFO)

class RegressionModel:
    
    @property
    def thetaHat(self):
        return self.thetaHat

    @property
    def plotColor(self):
        return self.plotColor
    
    def fit(self, x, y):
        logging.info('Regression model running...')

    def predict(self, x):
        logging.info('predicting...')
        
        

class LeastSquare(RegressionModel):
    
    def fit(self, x, y ):
        self.thetaHat = inv(x.dot(x.T)).dot(x).dot(y)
        print 'LeastSquare theta hat:',self.thetaHat

    def predict(self, x):
        return x.T.dot(self.thetaHat)
    
class RegularizedLeastSquare(RegressionModel):
    
    def fit(self, x, y, l ):
        self.thetaHat = inv(x.dot(x.T)+l).dot(x).dot(y)
        print 'RegularizedLeastSquare theta hat:',self.thetaHat

    def predict(self, x):
        return x.T.dot(self.thetaHat)

class Lasso(RegressionModel):
    
    def fit(self, x, y, l ):
        h_raw = x.dot(x.T)
        top = np.column_stack((h_raw, -h_raw))
        bottom = np.column_stack((-h_raw, h_raw))
        P = np.row_stack((top, bottom))
        y_raw = x.dot(y)
        q_raw = np.vstack((y_raw, -y_raw))
        q_raw.ravel()
        q_raw.shape = (2*x.shape[0],)
        q = l*np.array([1 for i in range(2*x.shape[0])]) - q_raw
        G = -1*np.eye(2*x.shape[0])
        h =  np.zeros((2*x.shape[0],1))
        logging.debug( 'P:%r', P.shape)
        logging.debug( 'q:%r', q.shape)
        logging.debug( 'G:%r', G.shape)
        logging.debug( 'h:%r', h.shape)
        sol = solvers.qp(matrix(P, tc='d'),matrix(q, tc='d'),matrix(G, tc='d'),matrix(h, tc='d'))
        combine = np.array(sol['x'])
        combine.ravel()
        combine.shape = (2, x.shape[0])
        self.thetaHat =  (combine[0]-combine[1]).T
        print 'Lasso theta hat: ',self.thetaHat
    
    def predict(self, x):
        return x.T.dot(self.thetaHat)

class RobustRegression(RegressionModel):
    
    def fit(self, x, y):
        D = x.shape[0]
        n = x.shape[1]
        zero_d = [0 for i in range(D)]
        one_n = [0 for i in range(n)]
        c = matrix(np.array(zero_d+one_n), tc='d')
        top = np.column_stack((-x.T, -np.identity(n)))
        bottom = np.column_stack((x.T, -np.identity(n)))
        A = matrix(np.row_stack((top, bottom)), tc='d')
        raw_y = np.row_stack((-y, y))
        raw_y.ravel()
        raw_y.shape = (2*n,)
        b = matrix(raw_y, tc='d')
        sol = solvers.lp(c, A, b)
        combine = np.array(sol['x'])
        combine.ravel()
        self.thetaHat =  combine[0:D]
        print 'RobustRegression theta hat:',self.thetaHat

    def predict(self, x):
        return x.T.dot(self.thetaHat)

    
class BayesianRegression(RegressionModel):
    
    @property
    def sigma_theta(self):
        return self.sigma_theta
    
    @property
    def mu_theta(self):
        return self.sigma_theta
        
    @property
    def beta(self):
        return self.beta
            
    @property
    def alpha(self):
        return self.alpha
    
    def fit(self, x, y ,alpha=0.48 , beta=1 ):
        self.alpha = alpha
        self.beta = beta
        self.sigma_theta = inv((1/alpha) * np.identity(x.shape[0]) + (1/beta) * np.dot(x, x.T))
        self.mu_theta = (1/beta )* np.dot(self.sigma_theta, np.dot(x, y))
        print 'BayesianRegression theta hat:',self.mu_theta
    
    def predict(self, x):
        return x.T.dot(self.mu_theta)
    
    def prediction_limit(self, x, stdevs = 1):
#         std_dev = x.T.dot(self.sigma_theta).dot(x)
        dim = x.shape[0]
        N = x.shape[1]
        m_x = x.reshape((dim, 1, N))
        predictions = []
        for idx in range(N):
            t_x = m_x[:,:,idx]
            sig_sq_x = 1/self.beta + t_x.T.dot(self.sigma_theta.dot(t_x))
            mean_x = t_x.T.dot(self.mu_theta)
            predictions.append((mean_x+stdevs*np.sqrt(sig_sq_x)).flatten())
        return np.concatenate(predictions)
        
def getMSE(trueY, predictY):
    return np.square(trueY - predictY).mean()

def getMAE(trueY, predictY):
    return np.abs(trueY - predictY).mean()

def show(x_poly, poly_Y, predict_Y, color, label, plotFlag =False):
    mse = getMSE(poly_Y, predict_Y.flatten())
    _label = label+", MSE:"+ str(mse)
    if plotFlag:
        plt.plot(x_poly, predict_Y, color = color, label=_label)
    return mse

def showCount(x_poly, poly_Y, predict_Y, color, label, plotFlag =False):
    mse = getMSE(poly_Y, predict_Y.flatten())
    mae = getMAE(poly_Y, predict_Y.flatten())
    _label = label+", MSE:"+ str(mse)+' , MAE:'+ str(mae)
    if plotFlag:
#         plt.plot(range(x_poly.shape[1]), poly_Y, c='xkcd:light '+color, label=_label+'-true')
        plt.plot(range(x_poly.shape[1]), predict_Y, color=color, label=_label+'-predict')
    return mse


def generatorOrderList(arr, order):
    matrix = []
    for i in range(0,order+1):
        matrix.append(map(lambda x: pow(x, i), arr))
    return matrix


def train(x_samples, y_samples, x_poly, y_poly, order , plotFlag = False):
    x_samples = np.array(x_samples)
    y_samples = np.array(y_samples)

    
    MSE = []
    
    mat_x =  np.array(generatorOrderList(x_samples, order))
    mat_x_test =  np.array(generatorOrderList(x_poly, order))
    logging.debug(mat_x)
    
    #Least square part
    LS = LeastSquare()
    try:
        LS.fit(mat_x, y_samples.T)
        LS_predictY = LS.predict(mat_x_test)
        
        MSE.append(show(x_poly, y_poly, LS_predictY , color = 'red', label=r'Least square fit', plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("LeastSquare fitting failed")
        MSE.append(None)
    finally:
        pass
        
    #Regularized Least Square
    dim = 5+1
    RegularizedLambda = 1
    RLS = RegularizedLeastSquare()
    try:
        RLS.fit(mat_x, y_samples.T, np.eye(order+1)*RegularizedLambda)
        MSE.append(show(x_poly, y_poly, RLS.predict(mat_x_test), color = 'blue', label='Regularized least square fit',  plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("RegularizedLeastSquare fitting failed")
        MSE.append(None)
    finally:
        pass
        
    Las = Lasso()
    lambd = 1
    try:
        Las.fit(mat_x, y_samples, lambd)
        Las_predictY = Las.predict(mat_x_test)
        MSE.append(show(x_poly, y_poly,  Las_predictY, color = 'yellow', label='LASSO fit',  plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("Lasso fitting failed")
        MSE.append(None)
    finally:
        pass
      
    #Robust Regression part
    RR = RobustRegression()
    try:
        RR.fit(mat_x, y_samples)
        RR_predictY = RR.predict(mat_x_test)
        MSE.append(show(x_poly, y_poly, RR_predictY, color = 'purple', label='Robust Regression fit',  plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("RobustRegression fitting failed")
        MSE.append(None)
    finally:
        pass
        
        
    BR = BayesianRegression()
    alpha = 1
    beta = 1/0.5
    try:
        BR.fit(mat_x, y_samples, alpha, beta)
        BR_predictY = BR.predict(mat_x_test)
        MSE.append(show(x_poly, y_poly, BR_predictY, color = 'pink', label='Bayesian Regression fit',  plotFlag = plotFlag))
        y_upper = BR.prediction_limit(mat_x_test, 1)
        y_lower = BR.prediction_limit(mat_x_test, -1)
        if plotFlag:
            plt.fill_between(x_poly,y_lower,y_upper,color='pink',alpha=0.5, label='BR standard deviation')
    except Exception as err:
        print(err)
        print("BayesianRegression fitting failed")
        MSE.append(None)
    finally:
        pass
    
    return MSE

def findBestAlphaAndBeta(x_samples, y_samples, x_poly, y_poly, order , plotFlag = False):
    alpha = 0.01
    beta = 1
    delta = 0.01
    mat_x =  np.array(generatorOrderList(x_samples, order))
    mat_x_test =  np.array(generatorOrderList(x_poly, order))
    currentMSE = 2000
    for i in range(10000):
        BR = BayesianRegression()
        BR.fit(mat_x, y_samples, alpha, beta)
        BR_predictY = BR.predict(mat_x_test)
        tmpMSE = show(x_poly, y_poly, BR_predictY, color = 'pink', label='Bayesian Regression fit',  plotFlag = False)
        if tmpMSE<currentMSE:
            currentMSE = tmpMSE
            print 'tmpMSE,a,b:', currentMSE, alpha, beta 
        
        alpha = alpha + delta
        
        
    

def plotErrorOfTrianSize(x_samples_all, y_samples_all, x_poly, y_poly, order):
    size_of_data = map(lambda x:int(x*len(x_samples_all)),[0.1,0.2, 0.3, 0.40, 0.5, 0.6, 0.7, 0.8 ,0.9 ,1.0])
    times = 10
    
    ls_mse = []
    rls_mse = []
    lasso_mse = []
    rr_mse = []
    br_mse = []
    
    for size in size_of_data:
        ls_mse_tmp = []
        rls_mse_tmp = []
        lasso_mse_tmp = []
        rr_mse_tmp = []
        br_mse_tmp = []
        for i in range(times):
            x_samples = random.sample(x_samples_all, size)
            y_samples = random.sample(y_samples_all, size)
            arr = train(x_samples, y_samples, x_poly, y_poly, order)
            if  arr[0]:
                ls_mse_tmp.append(arr[0])
            if  arr[1]:
                rls_mse_tmp.append(arr[1])
            if  arr[2]:    
                lasso_mse_tmp.append(arr[2])
            if  arr[3]: 
                rr_mse_tmp.append(arr[3])
            if  arr[4]: 
                br_mse_tmp.append(arr[4])
                
        if len(ls_mse_tmp) != 0:
            ls_mse.append((size ,sum(ls_mse_tmp)/len(ls_mse_tmp)))
        if len(rls_mse_tmp) != 0:
            rls_mse.append((size ,sum(rls_mse_tmp)/len(rls_mse_tmp)))
        if len(lasso_mse_tmp) != 0:
            lasso_mse.append((size ,sum(lasso_mse_tmp)/len(lasso_mse_tmp)))
        if len(rr_mse_tmp) != 0:
            rr_mse.append((size ,sum(rr_mse_tmp)/len(rr_mse_tmp)))
        if len(br_mse_tmp) != 0:
            br_mse.append((size ,sum(br_mse_tmp)/len(br_mse_tmp)))
    
    plt.grid()
    plotMSE(ls_mse , 'red', 'ls_mse ')
    plotMSE(rls_mse , 'yellow', 'rls_mse ')
    plotMSE(lasso_mse , 'green', 'lasso_mse ')
    plotMSE(rr_mse , 'blue', 'rr_mse ')
    plotMSE(br_mse , 'purple', 'br_mse ')
    
    plt.legend()
    plt.show() 
        
def plotMSE(mse, color,  _label):
    plt.ylim(0,500)  
    plt.plot(np.array(mse)[:,0], np.array(mse)[:,1], c= color, label=_label )

    
def plotBestFitOfAllData(x_samples, y_samples, x_poly, y_poly, order, plotFlag= True):  
    """
    TRUE THETA
      1.1524348407742859e+000    
      1.4862926001217334e+000    
      9.2950599238515164e-001    
     -1.1134419144922862e+000    
      1.5980451015263838e-001    
     -6.1788007767590925e-001
    """
    train(x_samples, y_samples, x_poly, y_poly, order, plotFlag= True) 
    plt.title("Polynomial function regression")
    plt.grid()
    plt.plot(x_poly, y_poly, c='black', label='true function')
    plt.scatter(x_samples, y_samples, s=20, c='green', label='sample')
    plt.legend()
    plt.show() 
    
def plotFitOfOutlierData(x_samples, y_samples, x_poly, y_poly, order, plotFlag= True):  
    plt.title("Polynomial function regression")
    plt.grid()
    plt.plot(x_poly, y_poly, c='black', label='poly data')
    
    outlier_x = [-1.01, -1.02, -1.03]
    outlier_y = map(lambda x:40+x,outlier_x)
    
    x = list(x_samples)
    y = list(y_samples)
    
    x.extend(outlier_x)
    y.extend(outlier_y)
    
    plt.scatter(x, y, s=40, c='red', marker = '3', label='outlier')
    plt.scatter(x_samples, y_samples, s=20, c='green', label='sample')
    train(x, y, x_poly, y_poly, order, plotFlag= True) 
    plt.legend()
    plt.show() 


def real_function(a_0, a_1, noise_sigma, x, covs=[1]):
    """
    Evaluates the real function
    """
    N = len(x)
    tmpSum = 0 
    for i in range(len(covs)):
        tmpSum = tmpSum + covs[i]*pow(x,i)
    if noise_sigma==0:
        # Recovers the true function
        return tmpSum
    else:
        return tmpSum + normal(0, noise_sigma, N)

    
    
def testFitOfHighOrderData():  
    
    testSize = 5000
    order = 10
    covsSet = [[1.1,0,0,0,0,-1.89,0,0,9.1,0],[0.03,3.41,-1.8,3,0.091,5,30.0,3.12,-0.02,1.2],[30,341,-1132,322,91,5231,30765,388,1344,87]]
    sizes = [10,100,1000]
    sigmas = [0.01,10,500]
    np.random.seed(20) # Set the seed so we can get reproducible results
    x_poly = np.array(sorted(uniform(-1, 1, testSize)))
    modelDict = {0:'LS', 1:'RLS' ,2:'LASSO' ,3:'RR' ,4:'BR'}
    with open('./outcome.txt', 'w') as f:
        tempWrite = []
        for covs in covsSet:
            for size in sizes:
                for sigma in sigmas:
                    tempWrite.append('\r\n==============paras=============\r\n')
                    tempWrite.append(' covs:'+str(covs))
                    tempWrite.append(' size:'+str(size))
                    tempWrite.append(' sigma:'+str(sigma))
                    
                    y_poly = real_function(4, 6, 0, x_poly,covs)  
                    x_samples = np.array(sorted(uniform(-1, 1, size)))
                    y_samples =  real_function(4, 6, sigma, x_samples,covs )  
                    out = train(x_samples, y_samples, x_poly, y_poly, order-1, plotFlag= False)
                    out =  [elem for elem in out if elem != None]
                    maxELe = np.max(out)
                    minELe = np.min(out)
                    tempWrite.append('\r\n WORST:'+str(modelDict.get(out.index(maxELe)))+' ,MSE:'+str(maxELe))
                    tempWrite.append('\r\n BEST:'+str(modelDict.get(out.index(minELe)))+' ,MSE:'+str(minELe))
        f.writelines(tempWrite)
    print 'complete!'
        
    
def plotFitOfHighOrderData():  
    plt.title("Polynomial function regression")
    plt.grid()
    
    size = 5000
    testSize = 100
    order = 10
    covs = [0.03,3.41,-1.8,3,0.091,5,30.0,3.12,-0.02,1.2]
    # Generate input features from uniform distribution
    np.random.seed(20) # Set the seed so we can get reproducible results
    x_poly = np.array(sorted(uniform(-1, 1, testSize)))
    
    
    # Evaluate the real function for training example inputs
    y_poly = real_function(4, 6, 0, x_poly,covs)  
   
    x_samples = np.array(sorted(uniform(-1, 1, size)))
    sigma = 20
    y_samples =  real_function(4, 6, sigma, x_samples,covs )  
    
    plt.plot(x_poly, y_poly, c='black', label='poly data')
    
    plt.scatter(x_samples, y_samples, s=1, c='green', label='sample')
    train(x_samples, y_samples, x_poly, y_poly, order-1, plotFlag= True) 
    plt.legend()
    plt.show() 
    

def generatorFeatureMatrix(arr):
    matrix = []
    matrix.append(arr)
    quadraticPart = map(lambda x: pow(x, 7), arr)
    matrix.append(quadraticPart)
    return np.array(matrix).reshape(2*len(arr),len(arr[0]))


def predictPeopleCount():
    x_samples_all =  np.loadtxt('./PA-1-data-text/count_data_trainx.txt')
    logging.debug('x samples: %r',x_samples_all)
    y_samples_all = np.loadtxt('./PA-1-data-text/count_data_trainy.txt')
    logging.debug('y samples: %r',y_samples_all)
    x_poly =  np.loadtxt('./PA-1-data-text/count_data_testx.txt')
    logging.debug('x samples: %r',x_poly)
    y_poly = np.loadtxt('./PA-1-data-text/count_data_testy.txt')
    logging.debug('y samples: %r',y_poly)
    
    MSE = []
    plotFlag = True
    
#     mat_x = x_samples_all
    mat_x = generatorFeatureMatrix(x_samples_all)
    y_samples = y_samples_all
    
#     mat_x_test = x_poly
    mat_x_test = generatorFeatureMatrix(x_poly)
    dim = mat_x_test.shape[0]
    
    plt.plot(range(x_poly.shape[1]), y_poly, color = 'black', label='true')
    
    #Least square part
    LS = LeastSquare()
    try:
        LS.fit(mat_x, y_samples.T)
        LS_predictY = LS.predict(mat_x_test)
        
        MSE.append(showCount(x_poly, y_poly, LS_predictY , color = 'red', label=r'Least square fit', plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("LeastSquare fitting failed")
        MSE.append(None)
    finally:
        pass
        
    #Regularized Least Square
    RegularizedLambda = 1
    RLS = RegularizedLeastSquare()
    try:
        RLS.fit(mat_x, y_samples.T, np.eye(dim)*RegularizedLambda)
        MSE.append(showCount(x_poly, y_poly, RLS.predict(mat_x_test), color = 'blue', label='Regularized least square fit',  plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("RegularizedLeastSquare fitting failed")
        MSE.append(None)
    finally:
        pass
        
    Las = Lasso()
    lambd = 1
    try:
        Las.fit(mat_x, y_samples, lambd)
        Las_predictY = Las.predict(mat_x_test)
        MSE.append(showCount(x_poly, y_poly,  Las_predictY, color = 'yellow', label='LASSO fit',  plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("Lasso fitting failed")
        MSE.append(None)
    finally:
        pass
      
    #Robust Regression part
    RR = RobustRegression()
    try:
        RR.fit(mat_x, y_samples)
        RR_predictY = RR.predict(mat_x_test)
        MSE.append(showCount(x_poly, y_poly, RR_predictY, color = 'purple', label='Robust Regression fit',  plotFlag = plotFlag))
    except Exception as err:
        print(err)
        print("RobustRegression fitting failed")
        MSE.append(None)
    finally:
        pass
        
        
    BR = BayesianRegression()
    alpha = 1
    beta = 1/0.5
    try:
        BR.fit(mat_x, y_samples, alpha, beta)
        BR_predictY = BR.predict(mat_x_test)
        MSE.append(showCount(x_poly, y_poly, BR_predictY, color = 'pink', label='Bayesian Regression fit',  plotFlag = plotFlag))
        y_upper = BR.prediction_limit(mat_x_test, 1)
        y_lower = BR.prediction_limit(mat_x_test, -1)
        if plotFlag:
            plt.fill_between(range(x_poly.shape[1]),y_lower,y_upper,color='green',alpha=0.1, label='BR standard deviation')
            
    except Exception as err:
        print(err)
        print("BayesianRegression fitting failed")
        MSE.append(None)
    finally:
        pass
    
    plt.legend()
    plt.show() 
    return MSE


def testDifferentRegressionModels():
    x_samples_all =  np.loadtxt('./PA-1-data-text/polydata_data_sampx.txt')
    logging.debug('x samples: %r',x_samples_all)
    y_samples_all = np.loadtxt('./PA-1-data-text/polydata_data_sampy.txt')
    logging.debug('y samples: %r',y_samples_all)
    x_poly =  np.loadtxt('./PA-1-data-text/polydata_data_polyx.txt')
    logging.debug('x samples: %r',x_poly)
    y_poly = np.loadtxt('./PA-1-data-text/polydata_data_polyy.txt')
    logging.debug('y samples: %r',y_poly)
    
    order = 5
#     plotErrorOfTrianSize(x_samples_all, y_samples_all, x_poly, y_poly, order )
#     plotBestFitOfAllData(x_samples_all, y_samples_all, x_poly, y_poly, order, plotFlag= True)
#     plotFitOfOutlierData(x_samples_all, y_samples_all, x_poly, y_poly, order, plotFlag= True)
#     findBestAlphaAndBeta(x_samples_all, y_samples_all, x_poly, y_poly, order, plotFlag= True)
#     plotFitOfHighOrderData()
    testFitOfHighOrderData()
    
def main():
#     testDifferentRegressionModels()
    predictPeopleCount()
    pass
    

if __name__ == '__main__':
    main()




    
  
    


