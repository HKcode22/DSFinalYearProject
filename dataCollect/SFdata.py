
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import seaborn as sns

# dataset_id = "g8m3-pdis"  # Registered Business Locations
# url = f"https://data.sfgov.org/resource/{dataset_id}.json"

# # Query for tech businesses (using NAICS codes for Information sector)
# params = {
#     "$where": "naic_code LIKE '51%' AND location_start_date > '2018-01-01T00:00:00'",
#     "$limit": 10000
# }

# response = requests.get(
#     url,
#     headers={"X-App-Token": app_token},
#     params=params
# )

# # Convert to DataFrame
# if response.status_code == 200:
#     data = response.json()
#     df_businesses = pd.DataFrame(data)
    
#     # Save to CSV (optional)
#     df_businesses.to_csv('sf_tech_businesses.csv', index=False)
#     print(f"Retrieved {len(df_businesses)} records")
# else:
#     print(f"Error: {response.status_code}")
#     print(response.text)


# # Data Analysis


# # Basic data exploration
# print(df_businesses.info())
# print(df_businesses.describe())

# # Check for missing values
# missing_values = df_businesses.isnull().sum()
# print("Missing values by column:")
# print(missing_values[missing_values > 0])

# # Convert date columns to datetime
# date_columns = ['location_start_date', 'location_end_date']
# for col in date_columns:
#     if col in df_businesses.columns:
#         df_businesses[col] = pd.to_datetime(df_businesses[col])

# # Extract year from start date
# if 'location_start_date' in df_businesses.columns:
#     df_businesses['start_year'] = df_businesses['location_start_date'].dt.year

# # Create visualizations
# plt.figure(figsize=(12, 6))
# df_businesses['start_year'].value_counts().sort_index().plot(kind='bar')
# plt.title('Number of Tech Businesses by Starting Year')
# plt.xlabel('Year')
# plt.ylabel('Count')
# plt.savefig('business_by_year.png')
# plt.show()

