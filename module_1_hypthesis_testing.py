import seaborn as sns
from scipy import stats

# 1. Define the hypotheses
hypothesized_mean_age = 30

# 2. Load the data
df = sns.load_dataset('titanic')
age_data = df['age'].dropna()

# 3. The one-sample t-test
t_statistic, p_value = stats.ttest_1samp(a=age_data, popmean=hypothesized_mean_age)

# 4. Interpretation
print(f"Null Hypothesis: The true mean age is exactly {hypothesized_mean_age}.")
print(f"Our Sample's Mean Age: {age_data.mean():.2f}")
print(f"P-value from the test: {p_value:.4f}")
print("-" * 50)

# 5. if/else statement to automatically make a conclusion.
if p_value < 0.05:
    print("Conclusion: Reject the Null Hypothesis.")
    print("The result is statistically significant. The true mean age is likely not 30.")
else:
    print("Conclusion: Fail to Reject the Null Hypothesis.")
    print("The result is not statistically significant. We don't have enough evidence to say the true mean age is different from 30.")
