import numpy as np

def gradient_cal(x_axis,y_axis):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x_axis)
    l_rate = 0.05

    for i in range(iterations):
        y_predicted = m_curr * x_axis + b_curr
        cost = (1/n) * sum([val**2 for val in (y_axis - y_predicted)])
        mdiff = -(2/n)*sum(x_axis*(y_axis-y_predicted))
        bdiff = -(2/n)*sum(y_axis-y_predicted)
        m_curr = m_curr - l_rate * mdiff
        b_curr = b_curr - l_rate * bdiff
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))



x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
print(x)
print(y)
gradient_cal(x,y)