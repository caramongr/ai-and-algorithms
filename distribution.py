import numpy as np
from scipy.stats import norm
import  matplotlib.pyplot as plt
import pandas as pd
import scipy
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x))
plt.show()
plt.plot(x, norm.cdf(x))
plt.show()
data = norm.rvs(size=10000, random_state=42)
plt.hist(data, bins=30)
plt.show()
x = np.linspace(12, 16, 100)
plt.plot(x, norm.pdf(x, loc=14, scale=0.5))
plt.show()
solar_data = norm.rvs(size=10000, loc=14, scale=0.5, random_state=42)
print(solar_data.mean())
print(solar_data.std())
df = pd.read_csv('data/solar_cell_efficiencies.csv')
df.describe()
# scipy.stats.norm.fit(df['efficiency'])
binom_dist = scipy.stats.binom(p=0.7, n=10)
plt.bar(range(11), binom_dist.pmf(k=range(11)))
plt.show()
print("======")
bs.bootstrap(df['efficiency'].values, stat_func=bs_stats.mean)