import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml 
import matplotlib.pyplot as plt

X = [[1,2],
     [1,4]]
Y = [2,3]
plt.scatter(la.transpose(X)[1],Y)

mlls = ml.Least_Squares(fit_intercept=False)
mlls.fit(X, Y)

print(mlls.coefs)

XLS = [[1,0],
       [1,1],
       [1,2],
       [1,3],
       [1,4],
       [1,5]]
YLS = mlls.predict(XLS)

plt.plot(la.transpose(XLS)[1], YLS)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
