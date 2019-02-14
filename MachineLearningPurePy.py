import LinearAlgebraPurePython as la 


class Least_Squares:
    def __init__(self, fit_intercept=True, tol=0, add_ones_column=True):
        """
        class structure for least squares regression 
        without machine learning libraries
            :param fit_intercept=True: If the data that 
                you send in does not have a column of 1's
                for fitting the y-intercept, it will be
                added by default for you. If your data has
                a column of 1's, set this parameter to False
            :param tol=0: This is a tolerance check used in
                the solve equations function in the
                LinearAlgebraPurePython module
        """
        self.fit_intercept = fit_intercept
        self.add_ones_column = add_ones_column
        self.tol = tol

    def __format_data_correctly(self, D):
        """
        Private function used to make sure data is formatted
        as needed by the various functions in the procedure;
        this allows more flexible entry of data
            :param D: The data structure to be formatted;
                assures the data is a list of lists
            :returns: Correctly formatted data
        """   
        if not isinstance(D[0],list): 
            return [D]
        else:
            return D

    def __orient_data_correctly(self, D):
        """
        Private function to ensure data is oriented 
        correctly for least squares operations;
        This allows more flexible entry of data
            :param D: The data structure to be 
                oriented; want more rows than columns
            :returns: Correctly oriented data
        """
        if len(D) < len(D[0]): 
            return la.transpose(D)
        else:
            return D

    def __condition_data(self, D):
        """
        Private function to format data in 
        accordance with the previous two private functions
            :param D: The data
            :returns: Correctly conditioned data
        """
        D = self.__format_data_correctly(D)
        D = self.__orient_data_correctly(D)

        return D

    def __add_ones_column_for_intercept(self, X):
        """
        Private function to append a column of 1's
        to the input matrix
            :param X: The matrix of input data
            :returns: The input matrix with a column
                of 1's appended to the front of it
        """   
        for i in range(len(X)): 
            X[i] = [1.0] + X[i]

        return X

    def fit(self, X, Y):
        """
        Callable method of an instance of this class
        to determine a set of coefficients for the 
        given data
            :param X: The conditioned input data
            :param Y: The conditioned output data
        """
        # Section 1: Condition the input and output data
        self.X = self.__condition_data(X)
        self.Y = self.__condition_data(Y)

        # Section 2: Append a column of 1's unless the 
        #     the user knows this is NOT necessary
        if self.fit_intercept and self.add_ones_column:
            self.X = self.__add_ones_column_for_intercept(self.X)

        # Section 3: Transpose the data into the null 
        #     space of the X matrix using the transpose of X
        #     and solve for the coefficients using our general 
        #     solve equations function
        AT = la.transpose(self.X)
        ATA = la.matrix_multiply(AT, self.X)
        ATB = la.matrix_multiply(AT, self.Y)
        self.coefs = la.solve_equations(ATA,ATB,tol=self.tol)

    def predict(self, X_test):
        """
        Condition the test data and then use the existing 
        model (existing coefficients) to solve for test
        results
            :param X_test: Input test data
            :returns:  Output results for test data
        """   
        # Section 1: Condition the input data
        self.X_test = self.__condition_data(X_test)

        # Section 2: Append a column of 1's unless the 
        #     the user knows this is NOT necessary
        if self.fit_intercept and self.add_ones_column:
            self.X_test = self.__add_ones_column_for_intercept(
                self.X_test)

        # Section 3: Apply the conditioned input data to the 
        #     model coefficients
        return la.matrix_multiply(self.X_test, self.coefs)


class Poly_Features_Pure_Py:
    def __init__(self, order=2, interaction_only=False, include_bias=True):
        """
        Mimics sklearn's PolyFeatures class to create various orders and types
        of polynomial variables from an initial set of supplied variables.
            :param order: the order of polynomials to be used - default is 2
            :param interaction_only: this means that only those polynomials
                with interaction, and that would add up in total power to the 
                given order, will be created for the set of polynomials. The
                default value is false.
            :param include_bias: the bias term is the one constant term in the
                final polynomial series. The default is to include it - True.
                To NOT include it, set this to False.
        """       
        self.order = order
        self.interaction_only = interaction_only
        self.include_bias = include_bias
            
        # Called to make sure all parameters have been correctly set
        self.__check_for_param_errors__()
        
    def fit(self, X):
        """
        Based on parameters and values of X, determine a set of powers 
        for each variable in the given X. This function only finds a list
        of powers that are needed. Transform applies those powers to the 
        lists of variables provided.
            :param X: a list of lists containing the values for which to 
                create power for.
        """

        # Make sure X is of the correct format.
        self.__check_X_is_list_of_lists__(X)

        # Determine the number of variables and create the initial 
        #   form of the powers list
        self.vars = len(X[0])
        self.powers = [0]*self.vars

        # Establish parameters and call the routine that gets the different
        #   combinations of powers needed for the order.
        order = self.order
        powers = self.powers
        self.__get_powers_lists__(order=order, 
            var=1, 
            powers=powers, 
            powers_lists=set())

        # Once all power combinations have been found, sort them.
        self.powers_lists.sort(reverse=True)

        # Eliminate powers not needed based on initialization parameters.
        self.__modify_powers_lists__()

    def get_feature_names(self, default_names=[]):
        """
        Routine to present the powers obtained from fit in an algebraic text format
            :param default_names: If this list is not empty, the text provided will
                be used in place of the default style of x0, x1, x2, ... xn
        """

        # Section 1: creates default names if not provided.
        if len(default_names) == 0:
            for i in range(self.vars):
                default_names.append('x' + str(i))
        # if default names are provided, makes sure they are of correct form
        elif len(default_names) != self.vars:
            err_str = 'Provide exactly {} feature names.'.format(self.vars)
            raise ValueError(err_str)
        elif len(default_names) == self.vars:
            check = [x for x in default_names if type(x) == str]
            if len(check) != self.vars:
                err_str = 'All feature names must be type string.'
                raise ValueError(err_str)

        # Section 2: Creates the features names based on the 
        #   default base feature names. 
        feature_names = []
        for powers in self.powers_lists:
            prod = []
            for i in range(len(default_names)):
                if powers[i] == 0:
                    continue
                elif powers[i] == 1:
                    val = default_names[i]
                else: 
                    val = default_names[i] + '^' + str(powers[i])
                prod.append(val)
            if prod == []:
                prod = ['1']
            feature_names.append(' '.join(prod))
        
        return feature_names

    def transform(self, X):
        """
        Apply the lists of powers previously found from fit to the provided
        arrays of X values
            :param X: The provided array of lists of input / feature values
        """   

        # Make sure X is of the correct format.
        self.__check_X_is_list_of_lists__(X)

        # Apply the powers found previously in fit to X
        Xout = []
        for row in X:
            temp = []
            for powers in self.powers_lists:
                prod = 1
                for i in range(len(row)):
                    prod *= row[i] ** powers[i]
                temp.append(prod)
            Xout.append(temp)

        return Xout
        
    def fit_transform(self, X):
        """
        Simlpy calls fit and transform in one step for convenience.
            :param X: The provided array of lists of input / feature values
        """   
        # Make sure X is of the correct format.
        self.__check_X_is_list_of_lists__(X)

        self.fit(X)
        return self.transform(X)

    def get_params(self):
        """
        Simply collects and returns the current parameter values in a 
        dictionary format
        """
        tmp_dict = {'order':self.order,
                    'interaction_only':self.interaction_only,
                    'include_bias':self.include_bias}

        print(tmp_dict)

    def set_params(self, **kwargs):
        """
        Allows user to provide keyword argument inputs to change parameters.
            :param **kwargs: keyword argument pairs are converted to 
                dictionary format
        """   
        if 'order' in kwargs:
            self.order = kwargs['order']
        if 'interaction_only' in kwargs:
            self.interaction_only = kwargs['interaction_only']
        if 'include_bias' in kwargs:
            self.include_bias = kwargs['include_bias']
            
        # Called to make sure all parameters have been correctly set
        self.__check_for_param_errors__()

    def __get_powers_lists__(self, order=2, var=1, powers=[0,0], powers_lists=set()):
        """
        Called from fit to obtain a set of power arrays for all instances of all 
        features. 
            :param order: default of 2 and used to set highest order
            :param var: current feature variable being worked on 
            :param powers: the current state of the powers array that will be 
                added to the list of powers
            :param powers_lists: the full set of powers lists
        """
        for pow in range(order+1):
            powers[var-1] = pow
            if sum(powers) <= order:
                powers_lists.add(tuple(powers))
            if var < self.vars:
                self.__get_powers_lists__(order=order, 
                    var=var+1, 
                    powers=powers, 
                    powers_lists=powers_lists)

        # Convert all tuples to lists for operational convenience
        self.powers_lists = [list(x) for x in powers_lists]

    def __modify_powers_lists__(self):
        """
        A private method to modify the powers lists based on the input parameters
        """
        # If only interactive combinations are desired, eliminate those features 
        
        #   that aren't interactive
        if self.interaction_only == True:
            self.powers_lists = [
                x for x in self.powers_lists if sum(x) == self.order]
        
        # If no bias is desired, remove it
        if self.include_bias == False:
            try:
                self.powers_lists.remove([0]*self.vars)
            except:
                pass

    def __check_for_param_errors__(self):
        """
        Simple method to ensure input parameters are of the correct type
        """
        error_string = ''
        if type(self.order) != int:
            error_string += '"order" needs to be of type int. '
        if type(self.interaction_only) != bool:
            error_string += '"interaction_only" needs to be of type bool. '
        if type(self.include_bias) != bool:
            error_string += '"include_bias" needs to be of type bool. '

        if error_string != '':
            raise TypeError(error_string)

    def __check_X_is_list_of_lists__(self, X):
        """
        Simple method to make sure that X input is of the correct format
        """
        error_string = 'X must be a list of lists.'
        if type(X) != list:
            raise TypeError(error_string)
        if type(X[0]) != list:
            raise TypeError(error_string)
