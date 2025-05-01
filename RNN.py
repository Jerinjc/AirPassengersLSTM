import numpy as np
from sklearn.model_selection import train_test_split

class RNN:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.01):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.learning_rate=learning_rate

        #Initialize weights
        self.Wxh=np.random.randn(self.input_size,self.hidden_size) * 0.01
        self.Whh=np.random.randn(self.hidden_size,self.hidden_size) * 0.01
        self.Why=np.random.randn(self.hidden_size,self.output_size) * 0.01

        self.bh=np.zeros((1,self.hidden_size))
        self.by=np.zeros((1,self.output_size))

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1-x)

    def forward(self,X):
        self.h=np.zeros((X.shape[0],self.hidden_size))

        for t in range(X.shape[1]):
            self.h=np.tanh(np.dot(X[:,t,:],self.Wxh)+np.dot(self.h,self.Whh)+self.bh)

        self.y=self.sigmoid(np.dot(self.h,self.Why)+self.by)
        return self.y
    
    def backward(self,X,y):
        m=y.shape[0]

        error=self.y - y
        d_output=error*self.sigmoid_derivative(self.y)

        dWhy=np.dot(self.h.T,d_output)
        dby=np.sum(d_output,axis=0,keepdims=True)

        dWhh=np.zeros_like(self.Whh)
        dWxh=np.zeros_like(self.Wxh)
        dbh=np.zeros_like(self.bh)

        dh_next=np.zeros_like(self.h)

        for t in reversed(range(X.shape[1])):
            dh=np.dot(d_output,self.Why.T)+dh_next
            dh_raw=(1-self.h**2)*dh #tanh derivative

            dWhh+=np.dot(self.h.T,dh_raw)
            dWxh+=np.dot(X[:,t,:].T,dh_raw)
            dbh+=np.sum(dh_raw,axis=0,keepdims=True)

            dh_next=np.dot(dh_raw,self.Whh.T)

        #Update weights
        self.Why-=self.learning_rate*dWhy/m
        self.by-=self.learning_rate*dby/m
        self.Whh-=self.learning_rate*dWhh/m
        self.Wxh-=self.learning_rate*dWxh/m
        self.bh-=self.learning_rate*dbh/m

    def train(self,X,y,epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X,y)
            if epoch % 100==0:
                loss=np.mean((self.y-y)**2)
                print(f"Epoch {epoch},Loss: {loss:.6f}")
    
    def predict(self,X):
        return self.forward(X)
    

# Generate dataset
X=np.random.rand(500,3,1)
y=(X.sum(axis=1)>1.5).astype(int)

#Split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train RNN
rnn=RNN(input_size=1,hidden_size=5,output_size=1,learning_rate=0.01)
rnn.train(X_train,y_train,epochs=1000)