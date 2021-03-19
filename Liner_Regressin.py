import numpy as np

class Liner_Regressin:
    def __init__(self,coss_type,learning_rate,n_iterations):
        self.cossType=coss_type
        self.learnRate=learning_rate
        self.iter=n_iterations
        if coss_type=="Mean_cost_function":
            self.cost_function=self.Mean_cost_function
            self.gradweight=self.Mean_update_weights
        elif coss_type=="Least_cost_function":
            self.cost_function=self.Least_cost_function
            self.gradweight=self.Least_update_weights
        else:
            print("wrong cossfunction type!")
    def predict(self,Features):
        bias = np.ones(shape=(len(X),1))
        Features = np.append(bias, X, axis=1)
        return Features @ self.weight
    def fit(self,X,Y):
        bias = np.ones(shape=(len(X),1))
        Features = np.append(bias, X, axis=1)
        Weights = np.random.ranf([Features.shape[1],1])
        for i in range(self.iter):
            Weights = self.gradweight(Features, Y, Weights)
            cost = self.cost_function(Features, Y, Weights)
            if i%1000==0:
                print("{}\t{:0.8}\t{}".format(i,cost,Weights.T))
        self.weight=Weights
        
    def Mean_cost_function(self,X, Y, Weights):
        N = len(Y)
        Ypred = X@Weights #predict(X, Weights)
        sq_error = (Y - Ypred)**2
        return 1/(2*N) * sq_error.sum()
    # Least Squared
    def Least_cost_function(self,X, Y, Weights):
        Ypred = X@Weights
        sq_error = (Y - Ypred)**2
        return sq_error.sum()/2
    def Mean_update_weights(self,X, Y, Weights):
        N = len(Y)
        Ypred = X@Weights
        error = Ypred - Y
        grad = (error.T @ X).T
        newWeights = Weights - self.learnRate *(1/N) * grad
        return newWeights
    
    def Least_update_weights(self,X, Y, Weights):
        #N = len(Y)
        Ypred = X@Weights
        error = Ypred - Y
        grad = (error.T @ X).T
        newWeights = Weights - self.learnRate * grad
        return newWeights
    
    def get_w(self):
        return self.weight
        