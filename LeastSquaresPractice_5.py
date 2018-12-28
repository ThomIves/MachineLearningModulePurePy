import MachineLearningPurePy as ml 
import LinearAlgebraPurePython as la 
import conditioned_data as cd
import sys


# Import conditioned data
X_train = cd.X_train
Y_train = cd.Y_train
X_test  = cd.X_test
Y_test  = cd.Y_test

# Solve for coefficients
mlls = ml.Least_Squares()
mlls.fit(X_train, Y_train)
print('Pure Coefficients:')
print(mlls.coefs, '\n')

# Make a prediction 
YLS = mlls.predict(X_test)
YLST = la.transpose(YLS)[0]

# Look at our predictions and the actual values
print('PurePredictions:\n', YLST[0], '\n')

# Compare to sklearn 
SKLearnData = [103015.20159796, 132582.27760816, 132447.73845175, 
               71976.09851258, 178537.48221056, 116161.24230165, 
               67851.69209676, 98791.73374687, 113969.43533013, 
               167921.06569551]

print('Delta Between SKLearnPredictions and Pure Predictions:')
for i in range(len(SKLearnData)):
    delta = round(SKLearnData[i],6) - round(YLST[i],6)
    print('\tDelta for outputs {} is {}'.format(i, delta))
