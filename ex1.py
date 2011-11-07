from matplotlib.pyplot import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D

def warmUpExercise():
    """
    WARMUPEXERCISE Example function in python
    A = warmUpExercise() is an example function that returns the 5x5 identity matrix
    """

    A = zeros(shape=(5,5))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Return the 5x5 identity matrix
    #       In octave, we return values by defining which variables
    #       represent the return values (at the top of the file)
    #        and then set them accordingly.



    # ============================================================

    return(A)

def plotData(x,y):
    """
    PLOTDATA Plots the data points x and y into a new figure
        plotData(x,y) plots the data points and gives the figure axes labels of
        population and profit.
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    plot(x, y, 'rx')                            # Plot the data
    ylabel('Profit in $10,000s')                # Set the y-axis label
    xlabel('Population of City in 10,000s')     # Set the x-axis label

    # ============================================================

def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
         J = computeCost(X, y, theta) computes the cost of using theta as the
         parameter for linear regression to fit the data points in X and y
    """

    # Initialize some useful values
    m = len(y)

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.



    # ============================================================

    return(J)

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
        theta, _ = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y)
    J_history = zeros(num_iters, dtype='f')

    for iter in xrange(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #                theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.


        # ============================================================

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return(theta, J_history)

if __name__ == '__main__':
    """
    Scaffold of Machine Learning Online Class - Excercise 1:
    Linear Regression in Python

    Original in Octave.

      Instructions
      ------------

      This file contains code that helps you get started on the
      linear exercise. You will need to complete the following functions
      in this exericse:

         ex1.py

     x refers to the population size in 10,000s
     y refers to the profit in $10,000s

    """

    ## ==================== Part 1: Basic Function ====================

    print("Running warmUpExercise ...\n")
    print("5x5 Identity Matrix: \n")
    print(warmUpExercise())

    raw_input("Program paused. Press enter to continue.\n")

    ## ======================= Part 2: Plotting =======================

    print("Plotting Data ...\n")
    data = genfromtxt("ex1data1.txt", delimiter=",")
    X = data[:,0]; y = data[:,1];
    m = len(y) # numer of training examples

    # Plot Data
    # Note: You have to complete the code in plotData
    plotData(X, y)
    show()

    raw_input("Program paused. Press enter to continue.\n")

    ##  =================== Part 3: Gradient descent ===================

    print("Running Gradient Descent ...\n")

    X = column_stack((ones(m), X))      # Add a column of ones to x
    theta = zeros((1,2))             # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    computeCost(X, y, theta)

    # run gradient descent
    theta, _ = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print("Theta found by gradient descent: ") ,
    print("%f %f \n" % (theta[0][0], theta[0][1]))

    # Plot the linear fit
    hold(True)
    plot(X[:,1:2], y, 'rx')
    plot(X[:,1], dot(X, theta.T), '-')
    legend(("Training data", "Linear regression"))
    hold(False)
    show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = dot([1, 3.5], theta.T)
    print("For population = 35,000, we predict a profit of %f\n" % (predict1*10000))
    predict2 = dot([1, 7.0], theta.T)
    print("For population = 70,000, we predict a profit of %f\n" % (predict2*10000))

    raw_input("Program paused. Press enter to continue.\n")

    ## ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print("Visualizing J(theta_0, theta_1) ...\n")

    # Grid over which we will calculate J
    theta0_vals = linspace(-10, 10, 100)
    theta1_vals = linspace(-1, 4, 100)

    # Initialize J_vals to a matrix of 0's
    J_vals = zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in xrange(len(theta0_vals)):
        for j in xrange(len(theta1_vals)):
            t = r_[theta0_vals[i], theta1_vals[j]]
            J_vals[i,j] = computeCost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # Surface plot
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')

    theta0_vals, theta1_vals = meshgrid(theta0_vals, theta1_vals)
    xlabel(r'$\Theta_0$')
    ylabel(r'$\Theta_1$')

    # this needs some work
    R = np.sqrt(theta0_vals**2 + theta1_vals**2)
    surf = ax.plot_surface(theta1_vals, theta0_vals, R)
    show()

    # found gist 1327152
    contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    show()

