'''
This file uses a dataset of 100 job descriptions from kaggle and processes the job descriptions to gain insight into the market.
Contains a function for cleaning text called clean_text(str)
'''
import os
import re

import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Start")

### Download job descriptions if haven't already
rawdata_path = "C:\\Users\\jacis\\.cache\\kagglehub\\datasets\\ivankmk\\thousand-ml-jobs-in-usa\\versions\\1"

if not os.path.exists(rawdata_path):
    rawdata_path = kagglehub.dataset_download("ivankmk/thousand-ml-jobs-in-usa")
    print("Downloaded. Path to dataset files:", rawdata_path)

df = pd.read_csv(rawdata_path + "\\1000_ml_jobs_us.csv")
# df.info()
# print(df.iloc[0]["job_description_text"])


### Plot and save top locations and companies
ax = df['company_address_locality'].value_counts().head(10).plot(kind='bar', title='Top 10 Localities')
fig = ax.get_figure()
fig.savefig('jobs_TopLocations.png', bbox_inches='tight')
fig.clear()

ax = df['company_name'].value_counts().head(10).plot(kind='barh', title='Top 10 Hiring Companies')
fig.savefig('jobs_TopCompanies.png', bbox_inches='tight')


### Preprocess and clean
# Remove rows with null job descriptions
df.dropna(subset=['job_description_text'], inplace=True)

# Define a basic cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)            # Replace multiple spaces with single space
    return text.strip().lower()                 # Lowercase and trim

# Apply cleaning to job descriptions
df['cleaned_desc'] = df['job_description_text'].apply(clean_text)

# Preview cleaned descriptions
df[['job_description_text', 'cleaned_desc']].head(3)


### Create a TF-IDF vectorizer to extract important words
tfidf = TfidfVectorizer(max_features=20, stop_words='english')

# Fit the vectorizer on cleaned job descriptions
tfidf_matrix = tfidf.fit_transform(df['cleaned_desc'])

# Extract top terms
top_terms = tfidf.get_feature_names_out()
print("Top 20 TF-IDF Keywords in Job Descriptions:")
print(', '.join(top_terms))


### See if we can predict seniority level from job description
# Drop rows with missing seniority labels
ml_data = df.dropna(subset=['seniority_level'])

# Prepare features (TF-IDF) and target
vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
X = vectorizer.fit_transform(ml_data['cleaned_desc'])
y = ml_data['seniority_level']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title('Confusion Matrix: Seniority Level Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Prediction Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=y_pred, order=pd.Series(y_pred).value_counts().index, palette='Set2')
plt.title("Predicted Seniority Levels")
plt.xlabel("Seniority Level")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()