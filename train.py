import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('data/symptoms-disease-dataset.csv')

# Debug 
print("Columns: ", data.columns.to_list())

if "Disease" not in data.columns:
    print("Disease column not found in the dataset!")
    exit()

# Convert text data into numerical labels
symptoms = list(set(data.columns) - {"Disease"})
data[symptoms] = data[symptoms].apply(lambda x: pd.factorize(x)[0])

# Split the dataset into training and testing sets
x = data[symptoms]
y = data["Disease"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Save the model
joblib.dump(model, 'model/symptoms-disease-model.pkl')
 
print("Model trained successfully!")