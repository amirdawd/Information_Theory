import numpy as np


class InfoTheory:
    def Entropy(self, P):
        # Input P:
        # Matrix (2-dim array): Each row is a probability
        # distribution, calculate its entropy,
        # Row vector (1Xm matrix): The row is a probability
        # distribution, calculate its entropy,
        # Column vector (nX1 matrix): Derive the binary entropy
        # function for each entry,
        # Single value (1X1 matrix): Derive the binary entropy
        # function
        # Output:
        # array with entropies
        entropiList = []
        if P.shape == (5,1):
            for x in P:
                entropi = 0.0
                if x != 0: 
                    entropi = self.binary_entropy(x[0])
                else:
                    entropi = -1 * x[0]
                entropiList.append(entropi)
            return entropiList
        if P.shape == (3,4):
            return self.prob_dist_entropy(P)

    def MutualInformation(self, P):
        # Derive the mutual information I(X;Y)
        # Input P: P(X,Y)
        # Output: I(X;Y)
        p_x = np.array([np.sum(P, axis=1)])
        p_y = np.array([np.sum(P, axis=0)])
        # I (X;Y) = H(X) + H(Y) - H(X,Y)
        h_x = self.prob_dist_entropy(p_x)
        h_y = self.prob_dist_entropy(p_y)
        h_x_y = np.sum(self.prob_dist_entropy(P))
        mutual_info = np.sum(h_x[0] + h_y[0] - h_x_y)
        return mutual_info
        
    def binary_entropy(self, value):
        if value ==1: 
            return 0 
        else:
            return (-1* (value * np.log2(value))) -((1-value)*np.log2(1-value))
    
    def prob_dist_entropy(self, values):
        entropi = 0.0
        if len(values.shape) == 1:
            for x in values:
                if x != 0:
                    entropi += -1 * (x * np.log2(x))
                else: 
                    entropi += -1 * x
            return entropi
        else: 
            entropiList = []
            for x in range(len(values)):
                entropi = 0.0
                for value in values[x]:
                    if value != 0:
                        entropi += -1 * (value * np.log2(value))
                    else: 
                        entropi += -1 * value
                entropiList.append(entropi)
            return entropiList


        
if __name__ == "__main__":
    IT = InfoTheory()
    ### 1st test
    P1 = np.transpose(np.array([np.arange(0.0,1.1,0.25)]))# rowvector
    H1 = IT.Entropy(P1)
    print("H1 =", H1)
     ### 2nd test
    P2 = np.array([[0.3, 0.1, 0.3, 0.3],
    [0.4, 0.3, 0.2, 0.1],
    [0.8, 0.0, 0.2, 0.0]])
    H2 = IT.Entropy(P2)
    print("H2 =",H2)
    ### 3rd test
    P3 = np.array([[0, 3/4],[1/8, 1/8]])
    I3 = IT.MutualInformation(P3)
    print("I3 =",I3)
    ### 4th test
    P4 = np.array([[1/12, 1/6, 1/3],
    [1/4, 0, 1/6]])
    I4 = IT.MutualInformation(P4)
    print("I4 =",I4)
   
    
 
