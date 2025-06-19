import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    # Read CSV but drop unnamed extra columns
    df = pd.read_csv("spam.csv", encoding='latin-1')

    # Keep only v1 and v2 columns (label and message)
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']  # Rename columns
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    return df


# Load and prepare data
df = load_data()
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict test set for accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# 🖥️ Streamlit UI
st.title("📩 SMS Spam Filter")
st.write("Enter a message below to check if it's spam or not using a Naive Bayes model!")

# Input from user
user_input = st.text_input("📨 Type your message here:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]

    if prediction == 1:
        st.error("🚨 This message is **SPAM**!")
    else:
        st.success("✅ This message is **HAM** )(not spam).")
    # Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")  