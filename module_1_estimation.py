import seaborn as sns
import pandas as pd

population_df = sns.load_dataset('titanic')
population_fare = population_df['fare'].dropna()

# 1. Calculate the true population parameter
population_mean = population_fare.mean()
print(f"True Population Mean Fare (The Parameter): {population_mean:.2f}")
print("-" * 50)

# 2. Sample of n=50 passengers
sample_df = population_fare.sample(n=50, random_state=42)

# 3. Calculate the Sample estimate from the sample of 50
sample_mean = sample_df.mean()
print(f"Sample Mean Fare (Estimate from n=50): {sample_mean:.2f}")
print("-" * 50)

# 4. With a different random sample of n=50
sample_2_mean = population_fare.sample(n=50, random_state=101).mean()
print(f"A Second Sample's Mean (n=50): {sample_2_mean:.2f}")
print("-" * 50)

# 5. With a larger sample of n=300
large_sample_mean = population_fare.sample(n=300, random_state=1).mean()
print(f"LArge Sample Mean (Estimate from n=300): {large_sample_mean:.2f}")
