import MachineLearningPurePy as ml 
import matplotlib.pyplot as plt

X = [2,3,4,2,3,4]
Y = [1.8,2.3,2.8,2.2,2.7,3.2]
plt.scatter(X,Y)

mlls = ml.Least_Squares()
mlls.fit(X, Y)

print(mlls.coefs)

XLS = [0,1,2,3,4,5]
YLS = mlls.predict(XLS)

plt.plot(XLS, YLS)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
