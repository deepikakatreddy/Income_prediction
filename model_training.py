# Build the Classification models and compare the results
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Create objects of classification algorithms with default hyper-parameters
ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=15)
ModelGNB = GaussianNB()

MM = [ModelLR, ModelDC, ModelKNN, ModelGNB]

for model in MM:
    # Train the model on training dataset
    model.fit(x_train, y_train)

    # Predict the model with test dataset
    y_pred = model.predict(x_test)

    # Print the model name
    print('Model Name: ', model)

    # Confusion matrix in sklearn
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    # Actual values
    actual = y_test

    # Predicted values
    predicted = y_pred

    # Confusion matrix
    matrix = confusion_matrix(actual, predicted, labels=[1, 0])
    print('Confusion matrix : \n', matrix)

    # Classification report for precision, recall, f1-score, and accuracy
    C_Report = classification_report(actual, predicted, labels=[1, 0])
    print('Classification report : \n', C_Report)

    # Accuracy score
    ac_score = accuracy_score(actual, predicted)
    print("Accuracy of the model: ", ac_score)
    print("<========================================================>")
