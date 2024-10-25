import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from numpy import *
from numpy.random import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import dataset

def task_1():
    # 1-st pow
    x1 = np.array([2, 6, 9, 15])
    y1 = np.array([9.9, 0.1, 9.2, 4.7])
    A = np.vstack([x1, np.ones(len(x1))]).T
    print(A)
    m1, c = np.linalg.lstsq(A, y1, rcond=None)[0]
    print(m1, c)
    plt.plot(x1, y1, 'o', label='Исходные данные', markersize=10)
    plt.plot(x1, m1 * x1 + c, 'r', label='Линейная экстраполяция')
    plt.legend()
    plt.show()
    # 2-nd pow
    delta = 1.0
    x2 = linspace(-20, 20, 11)
    y2 = x2 ** 2 + delta * (rand(11) - 0.5)
    x2 += delta * (rand(11) - 0.5)
    m2 = vstack((x2 ** 2, x2, ones(11))).T
    s1 = np.linalg.lstsq(m2, y2, rcond=None)[0]
    x_interval = linspace(-20, 20, 101)
    plt.plot(x2, y2, 'D')
    plt.plot(x_interval, s1[0] * x_interval ** 2 + s1[1] * x_interval + s1[2], '-', lw=2)
    plt.grid()
    plt.show()
    print(x2, y2, sep='\n')
    # 3-d pow
    m3 = vstack((x2 ** 3, x2 ** 2, x2, ones(11))).T
    s2 = np.linalg.lstsq(m3, y2, rcond=None)[0]
    plt.plot(x2, y2, 'D')
    plt.plot(x_interval, s2[0] * x_interval ** 3 + s2[1] * x_interval ** 2 + s2[2] * x_interval + s2[3], '-', lw=3)
    plt.grid()
    plt.show()


def f1(x, b0, b1):
    return b0 + b1 * x


def task_2_1():
    beta = (1, 1.75)
    xdata = np.linspace(0, 5, 50)
    y = f1(xdata, *beta)
    ydata = y + 0.05 * np.random.randn(len(xdata))
    beta_opt, beta_cov = sp.optimize.curve_fit(f1, xdata, ydata)
    print('\n\n', beta_opt, sep='')
    lin_dev = sum(beta_cov[0])
    print(lin_dev)
    residuals = ydata - f1(xdata, *beta_opt)
    difference = sum(residuals ** 2)
    print(difference)
    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f1(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
    plt.show()


def f2(x, b0, b1, b2):
    return b0 + b1 * x + b2 * x * x


def task_2_2():
    beta = (1, 1.75, 1.5)
    xdata = np.linspace(0, 5, 50)
    y = f2(xdata, *beta)
    ydata = y + 0.05 * np.random.randn(len(xdata))
    beta_opt, beta_cov = sp.optimize.curve_fit(f2, xdata, ydata)
    print('\n\n', beta_opt, sep='')
    lin_dev = sum(beta_cov[0])
    print(lin_dev)
    residuals = ydata - f2(xdata, *beta_opt)
    difference = sum(residuals ** 2)
    print(difference)

    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f2(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
    plt.show()


def f3(x, b0, b1):
    return b0 + b1 * np.log(x)


def task_2_3():
    beta = (2, 4)
    xdata = np.linspace(1, 5, 50)
    y = f3(xdata, *beta)
    ydata = y + 0.05 * np.random.randn(len(xdata))
    beta_opt, beta_cov = sp.optimize.curve_fit(f3, xdata, ydata)
    print('\n\n', beta_opt, sep='')
    lin_dev = sum(beta_cov[0])
    print(lin_dev)
    residuals = ydata - f3(xdata, *beta_opt)
    difference = sum(residuals ** 2)
    print(difference)

    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f3(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
    plt.show()


def f4(x, b0, b1):
    return b0 * x ** b1


def task_2_4():
    beta = (5, 6)
    xdata = np.linspace(1, 5, 50)
    y = f4(xdata, *beta)
    ydata = y + 0.05 * np.random.randn(len(xdata))
    beta_opt, beta_cov = sp.optimize.curve_fit(f4, xdata, ydata)
    print('\n\n', beta_opt, sep='')
    lin_dev = sum(beta_cov[0])
    print(lin_dev)
    residuals = ydata - f4(xdata, *beta_opt)
    difference = sum(residuals ** 2)
    print(difference)

    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f4(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
    plt.show()

def task_3():
    my_dict = {
        "time": [0.50, 0.75, 1.00, 1.25, 1.50, 1.75,1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50,
                 4.75, 5.00, 5.50],
        "mark": [10, 22, 13, 43, 20, 22, 33, 50, 62, 48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}
    data = pd.DataFrame(my_dict)
    print(data.head(), data.shape, data.describe(), sep='\n')
    plt.scatter(data['time'], data['mark'], color='b', label='exam data')
    plt.xlabel("Hour")
    plt.ylabel("Mark")
    plt.show()

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    print(x, y, sep='\n')
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    print(regressor.intercept_, regressor.coef_, sep='\n')

    y_predicted = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predicted})
    print(df)

    df.plot(kind='bar')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    plt.scatter(X_test, Y_test, color='gray')
    plt.plot(X_test, y_predicted, color='red', linewidth=2)


'''task_1()
task_2_1()
task_2_2()
task_2_3()
task_2_4()'''
task_3()