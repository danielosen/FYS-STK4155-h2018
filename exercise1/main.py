
import numpy as np

def get_random_linear_data(n=100,a=5,b=0.1):
	x = np.random.randn(n,1)
	y = a*x*x + b*np.random.randn(n,1)
	return x,y

def get_mse(ytrue,ypred):
	assert(ypred.shape == ytrue.shape)
	mse = ((ypred-ytrue).T).dot((ypred-ytrue)) * 1.0 / np.max(ypred.shape)
	return mse

def get_rsquared(ytrue,ypred):
	assert(ypred.shape == ytrue.shape)
	rsquared = 1 - ((ypred-ytrue).T).dot(ypred-ytrue) / ((ypred - np.mean(ypred)).T).dot(ypred - np.mean(ypred))
	return rsquared

def get_second_order_fit(x,y):
	assert(x.shape == y.shape)
	x = np.hstack((np.ones(x.shape),x,x*x))
	b = (np.linalg.inv((x.T).dot(x))).dot((x.T).dot(y))
	return b,x.dot(b)

def solve():
	x,ytrue = get_random_linear_data()
	b,ypred = get_second_order_fit(x,ytrue)
	print("parameters:\n{}\nmse:\n{}\nrsquared:\n{}".format(b,get_mse(ytrue,ypred),get_rsquared(ytrue,ypred)))

if __name__ == '__main__':
	solve()
