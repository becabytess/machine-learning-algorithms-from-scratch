import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = []
        self.b = 0.2
        self.X = []
        self.y = []

    def fit(self,X,y,lr=0.01,epochs=60000):
        self.X = X
        self.y = y
        for i in range(epochs):
            if i == 0:
                self.weights = np.random.randn(X.shape[1])

            self.forward()
            self.backward(lr)
        
    def predict(self,x):
        return self.sigmoid(np.dot(x,self.weights) + self.b)
    
    def sigmoid(self,linear_outputs):
        return 1/(1+np.exp(-linear_outputs))
    def forward(self):
        output = np.dot(self.weights,self.X.T)
        return self.sigmoid(output)
    def loss(preds,targets):
        for i in range(leng(preds)):
            if preds[i] == 0 or preds[i] == 1:
                preds[i] += 0.0000000000001
        return -np.mean(targets*np.log(preds)+(1-targets)*np.log(1-preds))
    
        

        
    def backward(self,lr):
        gradients = np.dot((self.forward() - self.y),self.X) 
        b_grad = np.mean(self.forward() - self.y)
        
        gradients /= len(self.X)
        # print(gradients)
        for i in range(len(self.weights)):
            self.weights[i] -= lr*gradients[i]
        self.b -= lr*b_grad 


model = LogisticRegression()

#using the iris dataset from sklearn 
# Suggested code may be subject to a license. Learn more: ~LicenseLog:696582996.
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
# print(y)
#convert it to binary problem 
# Suggested code may be subject to a license. Learn more: ~LicenseLog:1090120039.
binary_y  = []
binary_x = []
for i in range(len(y)):
    if y[i]==0 or y[i]==1:
        binary_y.append(y[i])
        binary_x.append(X[i])
X = np.array(binary_x)
y = np.array(binary_y)
model.fit(X,y)


for i in range(len(X)):
    pred = model.predict(X[i])
    target = y[i]
    print(pred,target)



