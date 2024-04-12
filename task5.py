import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Load the data from the spam-data.csv file
data = pd.read_csv("spam-data.csv")

# Split the data into features (X) and target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define a function to extract email features and check for spam
def check_spam(email_text):
    # Extract features from the email text based on known patterns
    # Count the occurrence of 'http://' to determine the number of links
    num_links = email_text.count('http://')

    # Count the occurrence of capitalized words (assuming words with all caps are considered capitalized)
    num_capitalized = sum(1 for word in email_text.split() if word.isupper())

    # Count the occurrence of 'http://' followed by 'sangu_and_nku_cybersecurity.ge/'
    num_spam_words = email_text.count('http://sangu_and_nku_cybersecurity.ge/')

    # Count the total number of words in the email
    num_words = len(email_text.split())

    # Create a DataFrame with named columns for the input features
    input_features = pd.DataFrame({'Number of Words': [num_words],
                                   'Number of Links': [num_links],
                                   'Number of Capitalized Words': [num_capitalized],
                                   'Number of Spam Words': [num_spam_words]})

    # Predict whether the email is spam or not based on the extracted features
    if model.predict(input_features)[0] == 1:
        return "Spam"
    else:
        return "Not Spam"

# Read the contents of emails.txt and split them into individual emails
with open("emails.txt", "r") as file:
    emails_content = file.read().split("----------------\n")

# Check each email for spam and print the results
for i, email in enumerate(emails_content, start=1):
    result = check_spam(email)
    print(f"Email {i}: {result}")
