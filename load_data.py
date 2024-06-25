# Load the dataset
traindata1 = pd.read_csv("path/to/income_prediction.csv", header=0)

# Copy to backup file
traindata2 = traindata1.copy()

# Display first 5 records
print(traindata1.head(5))
