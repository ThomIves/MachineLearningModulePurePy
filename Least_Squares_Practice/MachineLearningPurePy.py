import LinearAlgebraPurePython as la 


class Least_Squares:
    def __init__(self, fit_intercept=True, tol=0):
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
        if self.fit_intercept:
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
        if self.fit_intercept:
            self.X_test = self.__add_ones_column_for_intercept(
                self.X_test)

        # Section 3: Apply the conditioned input data to the 
        #     model coefficients
        return la.matrix_multiply(self.X_test, self.coefs)
