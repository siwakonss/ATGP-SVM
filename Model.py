import numpy as np
import cvxpy as cp
import Loss_funtions
from sklearn.base import BaseEstimator
#from tqdm import tqdm

class Hinge_SVM(BaseEstimator):
    def __init__(self,  C=1000):
        self.C = C
        self.w = None
        self.b = None
    def fit(self, X, y):
        m, n = X.shape  
        w = cp.Variable(n)
        b = cp.Variable(1)
        xi = cp.Variable(m)
        diagonal_matrix_y = np.zeros((len(y), len(y)))
        np.fill_diagonal(diagonal_matrix_y, y)
        # Define the objective function
        objective = cp.Minimize((0.5 * cp.norm(w)**2) + self.C *( cp.sum(xi)) )


        constraints = [
            xi >= np.ones(m) - (np.diag(y) @ (( (X @ w ) + b))),
            xi >= np.zeros(m)
        ]

        # Create the problem and solve it
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        #problem.solve(solver=cp.CVXOPT, verbose=False)

        # Extract the optimized variables
        w = w.value
        b = b.value

        self.w = w
        self.b = b
        return self.w, self.b
    
    def predict(self, X):
        # Compute the decision function
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions
    
    def score(self, X, y):
        # Assuming your prediction returns the predicted labels
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        return accuracy
    
class PIN_SVM(BaseEstimator):
    def __init__(self, tau=0.1,  C=1000):
        self.C = C
        self.tau = tau
        self.w = None
        self.b = None
    def fit(self,X, y):
        m, n = X.shape  # Use X_train's shape, as it's the standardized version
        w = cp.Variable(n)
        b = cp.Variable(1)
        xi = cp.Variable(m)
        diagonal_matrix_y = np.zeros((len(y), len(y)))
        np.fill_diagonal(diagonal_matrix_y, y)
        objective = cp.Minimize((0.5 * cp.norm(w)**2) + self.C *( cp.sum(xi)) )

        # Define the constraints

        constraints = [
            xi >= np.ones(m) - (np.diag(y) @ (( (X @ w ) + b))),
            xi >= -self.tau * (np.ones(m) - (np.diag(y) @ ((X @ w + b)))),
            xi >= np.zeros(m)
        ]

        # Create the problem and solve it
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        #problem.solve(solver=cp.CVXOPT, verbose=False)

        # Extract the optimized variables
        w = w.value
        b = b.value
        self.w = w
        self.b = b
        return self.w, self.b
    
    def predict(self, X):
        # Compute the decision function
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions
    
    def score(self, X, y):
        # Assuming your prediction returns the predicted labels
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        return accuracy

class GPIN_SVM(BaseEstimator):
    def __init__(self, tau1=0.1, tau2 =0.1, epsilon1 = 0.1, epsilon2 =0.1, C=1000):
        self.C = C
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2 
        self.w = None
        self.b = None
        self.b = None
    def fit(self,X, y):
        m, n = X.shape  # Use X_train's shape, as it's the standardized version
        w = cp.Variable(n)
        b = cp.Variable(1)
        xi = cp.Variable(m)

        objective = cp.Minimize((0.5 * cp.norm(w)**2) + self.C *( cp.sum(xi)) )

        # Define the constraints


        constraints = [
                xi >= (self.tau1 * (np.ones(m) - (np.diag(y) @ (( (X @ w ) + b))))) - self.epsilon1,
                xi >= (-self.tau2 * (np.ones(m) - (np.diag(y) @ ((X @ w + b))))) -self.epsilon2,
                xi >= np.zeros(m)
            ]

        # Create the problem and solve it
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        #problem.solve(solver=cp.CVXOPT, verbose=False)

        # Extract the optimized variables
        w = w.value
        b = b.value
        self.w = w
        self.b = b
        return self.w, self.b
    
    def predict(self, X):
        # Compute the decision function
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions
    
    def score(self, X, y):
        # Assuming your prediction returns the predicted labels
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        return accuracy
    
class TPIN_SVM(BaseEstimator):
    def __init__(self, tau=0.1, alpha1=2,  alpha2=2 , C=1000, num_iterations=50):
        self.C = C
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.tau = tau
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape  # Use X_train's shape, as it's the standardized version
        w = np.zeros(n)
        b = np.zeros(1)
        #theta = np.random.rand(n)
        for _ in range(self.num_iterations):
            sum_subgradient_h_TPIN =   Loss_funtions.TPIN()
            gamma = sum_subgradient_h_TPIN.sum_subgradient_h(w, b, X, y, self.tau , self.alpha1, self.alpha2)
            #print('gamma', gamma, gamma.shape)
            w = cp.Variable(n)
            b = cp.Variable(1)
            xi = cp.Variable(m)
            #diagonal_matrix_y = np.zeros((len(y), len(y)))
            objective = cp.Minimize((0.5 * cp.norm(w)**2) + self.C *( cp.sum(xi)) - self.C *( cp.matmul(gamma[:-1], w)) - self.C * (gamma[-1] * b))

            constraints = [
                xi >= np.ones(m) - (np.diag(y) @ (( (X @ w ) + b))),
                xi >= -self.tau * (np.ones(m) - (np.diag(y) @ ((X @ w + b)))),
                xi >= np.zeros(m)
            ]

            # Create the problem and solve it
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            #problem.solve(solver=cp.CVXOPT, verbose=False)

            # Extract the optimized variables
            w = w.value
            b = b.value
        self.w = w
        self.b = b
        return self.w, self.b
 
    def predict(self, X):
        # Compute the decision function
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions
    
    def score(self, X, y):
        # Assuming your prediction returns the predicted labels
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        return accuracy

class ATGP_SVM(BaseEstimator):
    def __init__(self, tau1=0.1,  tau2 =0.1, epsilon1 = 0.1, epsilon2 = 0.1, alpha1=2, alpha2=2,  C=1, num_iterations=50):
        self.C = C
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape  # Use X_train's shape, as it's the standardized version
        #theta =np.ones(n)
        w = np.zeros(n)
        b = np.zeros(1)
        #theta = np.random.rand(n)
        for _ in range(self.num_iterations):
            sum_subgradient_h_ATGPIN =   Loss_funtions.ATGPIN()
            gamma = sum_subgradient_h_ATGPIN.sum_subgradient_h(w, b, X, y, self.tau1, self.tau2, self.alpha1, self.alpha2)
            #print('gamma', gamma, gamma.shape)
            w = cp.Variable(n)
            b = cp.Variable(1)
            xi = cp.Variable(m)
            objective = cp.Minimize((0.5 * cp.norm(w)**2) + self.C *( cp.sum(xi)) - self.C *( cp.matmul(gamma[:-1], w)) - self.C * (gamma[-1] * b))

            # Define the constraints

            constraints = [
                xi >= (self.tau1 * (np.ones(m) - (np.diag(y) @ (( (X @ w ) + b))))) - self.epsilon1,
                xi >= (-self.tau2 * (np.ones(m) - (np.diag(y) @ ((X @ w + b))))) -self.epsilon2,
                xi >= np.zeros(m)
            ]

            # Create the problem and solve it
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            #problem.solve(solver=cp.CVXOPT, verbose=False)

            # Extract the optimized variables
            w = w.value
            b = b.value
        self.w = w
        self.b = b
        return self.w, self.b
 
    def predict(self, X):
        # Compute the decision function
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions
    
    def score(self, X, y):
        # Assuming your prediction returns the predicted labels
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        return accuracy
 