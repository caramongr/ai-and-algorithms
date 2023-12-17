import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np

loan_df = pd.read_csv('data/loan_data.csv',
parse_dates=['DATE_OF_BIRTH', 'DISBURSAL_DATE'],
infer_datetime_format=True)
report = ProfileReport(loan_df.sample(10000, random_state=42))
#report.to_file('loan_df1.html')
loan_df.corr().loc['LOAN_DEFAULT'][:-1].plot.barh()
plt.tight_layout()

q3 = loan_df['DISBURSED_AMOUNT'].quantile(0.75)
q1 = loan_df['DISBURSED_AMOUNT'].quantile(0.25)
iqr = (q3 - q1)
outliers = np.where(loan_df['DISBURSED_AMOUNT'] > (q3 + 1.5 * iqr))[0]
print(1.5 * iqr + q3)