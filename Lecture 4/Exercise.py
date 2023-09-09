import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression


def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_cal(x_axis,y_axis):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x_axis)
    l_rate = 0.000199999
    old_cost = 0

    for i in range(iterations):
        y_predicted = m_curr * x_axis + b_curr
        cost = (1/n) * sum([val**2 for val in (y_axis - y_predicted)])
        mdiff = -(2/n)*sum(x_axis*(y_axis-y_predicted))
        bdiff = -(2/n)*sum(y_axis-y_predicted)
        m_curr = m_curr - l_rate * mdiff
        b_curr = b_curr - l_rate * bdiff
        if (math.isclose(old_cost,cost, rel_tol = 1e-20, abs_tol = 0)):
            break
        old_cost = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))

    return m_curr, b_curr

data = pd.read_csv("test_scores.csv")
x_arr = np.array(data.math)
y_arr = np.array(data.cs)

m,b = gradient_cal(x_arr,y_arr)
print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

m_sklearn, b_sklearn = predict_using_sklean()
print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))
