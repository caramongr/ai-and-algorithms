#import modules
import numpy as np
import matplotlib.pyplot as plt

#generate some data points
x = np.linspace(-5, 5, 100)
y = 2*x**3 - 5*x**2 + 3*x + 7 + np.random.randn(100)

#find the best polynomial fit of degree 3
coeffs = np.polyfit(x, y, 3)

#generate the fitted values
yfit = np.polyval(coeffs, x)
print(coeffs)

#plot the data points and the line of best fit
plt.scatter(x, y, label='data')
plt.plot(x, yfit, color='red', label='best fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()