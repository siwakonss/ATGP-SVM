import numpy as np



class ATGPIN:
    def __init__(self,tau1=0.1, tau2 =0.1, alpha1=2, alpha2=3, epsilon1 = 0.1, epsilon2= 0.1):
        self.tau1 = tau1
        self.tau2 = tau2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
    def g(self,u, tau1, tau2, epsilon1, epsilon2):
        if epsilon1 / tau1 <= u:
            return tau1 * ( u -(epsilon1/tau1))  
        elif -epsilon2 / tau2 <= u <= epsilon1 / tau1:
            return 0 
        elif u <=  -epsilon2 / tau2:
            return -tau2 * ( u +(epsilon2/tau2)) 

    def h(self,u, tau1, tau2, alpha1, alpha2):
        if alpha1 / tau1 <= u:
            return tau1 * ( u -(alpha1/tau1)) 
        elif -alpha2 / tau2 <= u <= alpha1/ tau1:
            return 0 
        elif u <=  -alpha2 / tau2:
            return -tau2* ( u +(alpha2/tau2)) 

    def sum_g(self,w, b, X, y, tau1, tau2, epsilon1, epsilon2):
        total_loss_g = 0
        N = X.shape[0]  
        #X = np.c_[X,np.ones(X.shape[0])] # Use X_train's shape, as it's the standardized version
        for i in range(N):
            u = 1 - ((y[i] * (np.dot(X[i], w))+b))
            g_value = ATGPIN.g(self,u, tau1, tau2, epsilon1, epsilon2)
            total_loss_g += g_value
        return total_loss_g

    def sum_h(self,w,b,X, y, tau1, tau2, alpha1, alpha2):
        total_loss_h = 0
        N = X.shape[0] 
       # X = np.c_[X, np.ones(X.shape[0])] # Use X_train's shape, as it's the standardized version
        for i in range(N):
            u = 1 - (y[i] * (np.dot(X[i], w)+b))
            h_value = ATGPIN.h(self,u, tau1, tau2, alpha1, alpha2)
            total_loss_h += h_value
        return total_loss_h

    def subgradient_h(self,u, X, y, tau1, tau2, alpha1, alpha2):
        value_to_add = 1
    # Add the value to the last column
        X1 = np.append(X, value_to_add)
        p = X1.shape[0]
        #print('p', p)
        #print('-tau2*(-y * X1)', -tau2*(-y * X1))
        #print('tau2*(y * X1)', tau2*(y * X1))
        #print(-tau2*(-y * X1) == tau2*(y * X1))
        if u > alpha1 / tau1:
            return -tau1*(y * X1)
        elif -(alpha2 / tau2) <= u and u <= alpha1/tau1:
            return  np.zeros(p)
        else:
            return tau2*(y * X1)

    def sum_subgradient_h(self,w, b, X, y, tau1, tau2, alpha1, alpha2):
        total_sgh = np.zeros(len(w)+1)
        N = X.shape[0]  # Use X_train's shape, as it's the standardized version
        for i in range(N):
            u = 1 - (y[i] * ((X[i] @ w )+b))
            subgradient_h_value = ATGPIN.subgradient_h(self,u,X[i],y[i], tau1, tau2, alpha1, alpha2)
            total_sgh += subgradient_h_value
        return total_sgh



class TPIN:
    def __init__(self,tau=0.1, alpha1=2, alpha2=3):
        self.tau = tau
        self.alpha1 = alpha1
        self.alpha2 = alpha2
 
    def g(self,u, tau):
        if 0 <= u:
            return u 
        elif u <= 0:
            return -tau * u 

    def h(self,u, tau, alpha1, alpha2):
        if alpha1 <= u:
            return u - alpha1 
        elif -(alpha2 / tau) <= u <= alpha1:
            return 0
        elif u <= -(alpha2 / tau) :
            return -tau * (u + (alpha2 / tau)) 

    def sum_g(self,w, b, X, y, tau):
        total_loss_g = 0
        N = X.shape[0]  
        #X = np.c_[X,np.ones(X.shape[0])] # Use X_train's shape, as it's the standardized version
        for i in range(N):
            u = 1 - ((y[i] * (np.dot(X[i], w))+b))
            g_value = TPIN.g(self,u, tau)
            total_loss_g += g_value
        return total_loss_g

    def sum_h(self,w,b,X, y, tau, alpha1, alpha2):
        total_loss_h = 0
        N = X.shape[0] 
       # X = np.c_[X, np.ones(X.shape[0])] # Use X_train's shape, as it's the standardized version
        for i in range(N):
            u = 1 - (y[i] * (np.dot(X[i], w)+b))
            h_value = TPIN.h(self,u, tau, alpha1, alpha2)
            total_loss_h += h_value
        return total_loss_h

    def subgradient_h(self,u, X, y, tau, alpha1, alpha2):
        value_to_add = 1
    # Add the value to the last column
        X1 = np.append(X, value_to_add)
        p = X1.shape[0]
        #print('p', p)
        if u > alpha1 :
            return -y * X1
        elif -(alpha2 / tau) <= u and u <= alpha1:
            return  np.zeros(p)
        else:
            return -tau*(-y * X1)

    def sum_subgradient_h(self,w, b, X, y, tau, alpha1, alpha2):
        total_sgh = np.zeros(len(w)+1)
        N = X.shape[0]  # Use X_train's shape, as it's the standardized version
        for i in range(N):
            u = 1 - (y[i] * ((X[i] @ w )+b))
            subgradient_h_value = TPIN.subgradient_h(self,u,X[i],y[i],tau, alpha1, alpha2)
            total_sgh += subgradient_h_value
        return total_sgh
    


    