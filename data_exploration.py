# Compute the correlation matrix
correlation_matrix = traindata2.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="Greens", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Create a count plot for income distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=traindata2, x='income_>50K')
plt.title("Income Distribution")
plt.xlabel("Income > $50K")
plt.ylabel("Count")
plt.show()
