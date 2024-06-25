# Display dataset information
traindata2.info()
print(traindata2.shape)
print(traindata2.describe())
print(traindata2.dtypes)

# Identify and remove duplicates
print(traindata2[traindata2.duplicated(keep='first')])
traindata2 = traindata2.drop_duplicates(keep='first')

# Resetting the index
traindata2.reset_index(inplace=True)

# Check for null values
print(traindata2.isnull().sum())

# Deleting index column as it has no influence on target variable
del traindata2['index']

# Fill missing values
traindata2 = traindata2.fillna(method='ffill')
print(traindata2.isnull().sum())

# Transform categorical columns to integer values using LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cols = ['workclass', 'occupation', 'native-country', 'marital-status', 'relationship', 'race', 'gender', 'education']
for col in cols:
    traindata2[col] = le.fit_transform(traindata2[col])

X = traindata2.iloc[:, :-1]
y = traindata2.iloc[:, -1]
