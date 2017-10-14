# LinearRegressionCompare
Implement and compare 5 different linear regression methods 
1. least-squares (LS)
2. regularized LS (RLS)
3. L1-regularized LS (LASSO)
4. robust regression (RR)
5. Bayesian regression (BR)

## There are several methods in the main() for different usage.

### testDifferentRegressionModels()
	1. plotErrorOfTrianSize(x_samples_all, y_samples_all, x_poly, y_poly, order )
   	2. plotBestFitOfAllData(x_samples_all, y_samples_all, x_poly, y_poly, order, plotFlag= True)
    3. plotFitOfOutlierData(x_samples_all, y_samples_all, x_poly, y_poly, order, plotFlag= True)
    4. findBestAlphaAndBeta(x_samples_all, y_samples_all, x_poly, y_poly, order, plotFlag= True)
    5. plotFitOfHighOrderData()
    6. testFitOfHighOrderData()
### predictPeopleCount()
