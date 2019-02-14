# PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
import sys


class PolynomialTools:
    def __init__(self, order=2, interactions_only=False, include_bias=True):
        self.order = order
        self.interactions_only = interactions_only
        self.include_bias = include_bias

    def __reduce_fit_to_interactions_only(self):
        new_list = []
        power_list_length = len(self.__powers_lists[0])
        for power_list in self.__powers_lists:
            if 1 <= power_list.count(0) < power_list_length:
                continue
            else:
                new_list.append(power_list)

        self.__powers_lists = new_list

    def __remove_bias(self):
        self.__powers_lists = [x for x in self.__powers_lists if sum(x) != 0]

    def fit(self, X):
        self.__vars = len(X[0])
        self.__powers = [0]*self.__vars
        self.__features_names = ['x'+str(i) for i in range(self.__vars)]

        self.get_powers_lists(
            order=self.order, 
            var=1, 
            powers=self.__powers, 
            powers_lists=set())

        if self.interactions_only:
            self.__reduce_fit_to_interactions_only()

        if not self.include_bias:
            self.__remove_bias()

    def get_powers_lists(self, 
        order=2, 
        var=1,
        powers=[0, 0],
        powers_lists=set()):

        for pow in range(order+1):
            powers[var-1] = pow
            if sum(powers) <= order:
                powers_lists.add(tuple(powers))
            if var < self.__vars:
                self.get_powers_lists(
                    order=order,
                    var=var+1,
                    powers=powers,
                    powers_lists=powers_lists)

        self.__powers_lists = [list(x) for x in powers_lists]
        self.__powers_lists.sort(reverse=True)

    def transform(self, X):
        self.X_poly = []
        for Xrow in X:
            self.X_poly.append([])
            for power_list in self.__powers_lists:
                val = 1.0
                for i in range(self.__vars):
                    val *= Xrow[i]**power_list[i]
                self.X_poly[-1].append(val)

    def fit_transform(self,X):
        self.fit(X)
        self.transform(X)

    def get_features_names(self, features_names=[]):
        if len(features_names) != 0:
            self.__features_names = features_names
        
        temp_features = []
        for power_list in self.__powers_lists:
            current_string = ''
            if sum(power_list) == 0:
                current_string = '1'
            else:
                for i in range(self.__vars):
                    if power_list[i] == 0:
                        continue
                    elif power_list[i] == 1:
                        current_string += \
                            self.__features_names[i] + ' '
                    else:
                        current_string += \
                            self.__features_names[i] + \
                            '^' + str(power_list[i]) + ' '
            temp_features.append(current_string.rstrip(' '))

        return temp_features 


X = [[1, 2], [3, 4], [5, 6]]

my_poly = PolynomialTools(order=2) #, interactions_only=True, include_bias=False)
my_poly.fit(X)

# print(len(my_poly.__powers_lists))
# print(my_poly.__powers_lists)
print(my_poly.get_features_names(['y0', 'y1']))
print()
sys.exit()

my_poly.transform(X)

for row in my_poly.X_poly:
    print(row)
print()

my_poly1 = PolynomialTools(order=2) #, interactions_only=True, include_bias=False)
my_poly1.fit_transform(X)

for row in my_poly1.X_poly:
    print(row)
