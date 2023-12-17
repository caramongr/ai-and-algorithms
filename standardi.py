import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer

loan_df = pd.read_csv('data/loan_data.csv',
parse_dates=['DATE_OF_BIRTH', 'DISBURSAL_DATE'],
infer_datetime_format=True)



pt = PowerTransformer()
loan_df['xform_age'] = pt.fit_transform(loan_df['AGE'].values.reshape(-1, 1))
f, ax = plt.subplots(2, 1)
loan_df['AGE'].plot.hist(ax=ax[0], title='AGE', bins=50)
loan_df['xform_age'].plot.hist(ax=ax[1], title='xform_age', bins=50)
plt.tight_layout()