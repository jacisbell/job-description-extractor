'''
This file uses a dataset of 100 job descriptions from kaggle and processes the job descriptions to gain insight into the market.
Contains a function for cleaning text called clean_text(str). 
Performs classification of seniority level based on job description
Generates images for top companies, top locations, and classification confusion matrix.

The confusion matrix shows how many descriptions for entry level positions look like mid-senior level descriptions.
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
from sklearn.metrics import confusion_matrix

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
fig.clear()


### Preprocess and clean
# Remove rows with null job descriptions
df.dropna(subset=['job_description_text'], inplace=True)

# Define a basic cleaning function, removing special characters, removing multiple spaces, lowercase and trim.
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

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
print(', '.join(top_terms) + "\n")

### See if we can predict seniority level from job description
# Drop rows with missing seniority labels or counts less than 50
ml_data = df.dropna(subset=['seniority_level'])
ml_data = ml_data[~ml_data['seniority_level'].str.contains('Not Applicable', na=False)]

v = ml_data['seniority_level'].value_counts()
# print(v)
ml_data = ml_data[ml_data['seniority_level'].isin(v.index[v.gt(5)])]
print(ml_data['seniority_level'].value_counts())


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
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title('Confusion Matrix: Seniority Level Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
fig = ax.get_figure()
fig.savefig('jobs_ConfusionMatrix.png', bbox_inches='tight')
fig.clear()