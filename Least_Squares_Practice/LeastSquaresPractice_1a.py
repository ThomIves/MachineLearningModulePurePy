import MachineLearningPurePy as ml 
import matplotlib.pyplot as plt

X = [2,4]
Y = [2,3]
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
