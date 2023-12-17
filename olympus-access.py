import pandas as pd

input_file = "input.xlsx"
df = pd.read_excel(input_file)


latest_access_dates = df.groupby('Name')['Last Access Date'].max().reset_index()


result_df = pd.merge(latest_access_dates, df, on=['Name', 'Last Access Date'], how='left')

output_file = "output.xlsx"
result_df.to_excel(output_file, index=False)

print("Data exported successfully to", output_file)
