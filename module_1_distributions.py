import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = sns.load_dataset('titanic')

# 2. Isolate the random variable
fare = df['fare'].dropna()

# 3. Summary statistics
print("Summary Statistics for Titanic Passenger Fare:")
print(fare.describe())

# 4. Visualize the distribution with a histogram
plt.figure(figsize=(10, 6))
plt.hist(fare, bins=50, edgecolor='black', color="#416baf")
plt.title('Distribution of Passenger Fare on the Titanic')
plt.xlabel('Fare (in contemporary currency)')
plt.ylabel('Frequency (Number of Passengers)')
plt.grid(axis='y', alpha=0.75)

plt.show()
