import pandas as pd
from scipy.stats import ttest_1samp
from scipy.stats import kstest, norm, skewnorm

solar_data = pd.read_csv('data/solar_cell_efficiencies.csv')
# print(ttest_1samp(solar_data['efficiency'], 14, alternative='two-sided'))


sample = solar_data['efficiency'].sample(30, random_state=1)
print(sample.mean())
print(ttest_1samp(sample, 14))

print(kstest(solar_data['efficiency'], norm(loc=14, scale=0.5).cdf))