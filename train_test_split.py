# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display size of train and test sets
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Scaling the features using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
