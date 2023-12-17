import pandas as pd

temps = {'Mon':[68,89],'Tue':[71,93],'Wed':[66,82],'Thu':[75,97],'Fri':[62,79]}

temperatures = pd.DataFrame(temps,index=['Low','High'])

print(temperatures)

print(temperatures.loc[:,'Mon':'Wed'])

print(temperatures.loc['Low'])


print(temperatures.iloc[1])


print(temperatures[0:1])

print(temperatures.describe()) #gives the mean, std, min, max, etc.
