from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np 


X = np.arange(6).reshape(3, 2)

print(X)
print()

poly1 = pf(2)

# Xnew1 = poly1.fit_transform(X)
# print(Xnew1)

feat_names = poly1.get_feature_names(['y0','y1'])
print(feat_names)

feat_poly  = poly1.get_params()
print(feat_poly)

# print()

# poly2 = pf(2,interaction_only=True)
# Xnew2 = poly2.fit_transform(X)

# print(Xnew2)
