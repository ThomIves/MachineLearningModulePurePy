import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

X = [[2,3,4,2,3,4],[1,2,3,1,2,3]]
Y = [1.8,2.3,2.8,2.2,2.7,3.2]
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(X[0], X[1], Y)
ax.set_xlabel('X1 Values')
ax.set_ylabel('X2 Values')
ax.set_zlabel('Y Values')
ax.set_title('Pure Python Least Squares Two Inputs Data Fit')

mlls = ml.Least_Squares()
mlls.fit(X, Y)

print(mlls.coefs)

XLS = [[1,1.5,2,2.5,3,3.5,4],[0,0.5,1,1.5,2,2.5,3]]
YLS = mlls.predict(XLS)
YLST = la.transpose(YLS)[0]

ax.plot3D(XLS[0], XLS[1], YLST)
plt.show()
