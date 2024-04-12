import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data from spam-data.csv
data = pd.read_csv("spam-data.csv")

# Split the data into features (X) and target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model on emails.txt
# For the first email:
# Number of Words: 36
# Number of Links: 3
# Number of Capitalized Words: 2
# Number of Spam Words: 0
# Therefore, the feature vector for this email is [36, 3, 2, 0]
test_email_features = [[36, 3, 2, 0]]

# Predict whether the email is spam or not
prediction = model.predict(test_email_features)

# Print the result
if prediction[0] == 1:
    print("The first email is classified as spam.")
else:
    print("The first email is not classified as spam.")