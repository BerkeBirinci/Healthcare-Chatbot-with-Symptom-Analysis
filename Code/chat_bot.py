import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the datasets
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

# Preprocess the data
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode target labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Function definitions
severityDictionary = {}
description_list = {}
precautionDictionary = {}

symptoms_present = []


def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                symptom, severity = parts
                severityDictionary[symptom] = int(severity)

def getDescription():
    with open('MasterData/symptom_Description.csv') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                symptom, description = parts
                description_list[symptom] = description

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) > 1:
                precautionDictionary[parts[0]] = parts[1:]

def calc_condition(symptoms, days):
    severity = sum(severityDictionary.get(symptom, 0) for symptom in symptoms)
    if (severity * days) / (len(symptoms) + 1) > 13:
        return "You should consult a doctor."
    else:
        return "It might not be serious, but take precautions."

def predict_disease(symptoms):
    input_vector = np.zeros(len(cols))
    for symptom in symptoms:
        if symptom in cols:
            input_vector[cols.get_loc(symptom)] = 1
    prediction = clf.predict([input_vector])[0]
    return le.inverse_transform([prediction])[0]

def ask_additional_symptoms(possible_symptoms):
    additional_symptoms = []
    for symptom in possible_symptoms:
        answer = messagebox.askyesno("Symptom Check", f"Are you experiencing {symptom}?")
        if answer:
            additional_symptoms.append(symptom)
    return additional_symptoms

# Load dictionaries
getSeverityDict()
getDescription()
getprecautionDict()

# GUI Application
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Healthcare Chatbot")

        # Input for symptoms
        self.symptom_label = tk.Label(root, text="Enter your symptoms (comma-separated):")
        self.symptom_label.pack(pady=5)

        self.symptom_entry = tk.Entry(root, width=50)
        self.symptom_entry.pack(pady=5)

        # Input for duration
        self.duration_label = tk.Label(root, text="How many days have you been experiencing them?")
        self.duration_label.pack(pady=5)

        self.duration_entry = tk.Entry(root, width=10)
        self.duration_entry.pack(pady=5)

        # Submit button
        self.submit_button = tk.Button(root, text="Submit", command=self.process_input)
        self.submit_button.pack(pady=10)

        # Output area
        self.output_text = tk.Text(root, height=15, width=60, state=tk.DISABLED)
        self.output_text.pack(pady=10)

    def process_input(self):
        symptoms = self.symptom_entry.get().split(',')
        days = self.duration_entry.get()

        try:
            days = int(days)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of days.")
            return

        # Predict initial disease
        disease = predict_disease(symptoms)
        possible_symptoms = cols[clf.feature_importances_ > 0.01]  # Example filtering for relevant symptoms
        additional_symptoms = ask_additional_symptoms(possible_symptoms)

        symptoms.extend(additional_symptoms)
        final_disease = predict_disease(symptoms)

        condition = calc_condition(symptoms, days)
        precautions = precautionDictionary.get(final_disease, [])
        description = description_list.get(final_disease, "No description available.")

        # Display results
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Predicted Disease: {final_disease}\n")
        self.output_text.insert(tk.END, f"Description: {description}\n")
        self.output_text.insert(tk.END, f"Condition Advice: {condition}\n")
        self.output_text.insert(tk.END, "Precautions:\n")
        for i, precaution in enumerate(precautions, 1):
            self.output_text.insert(tk.END, f"  {i}. {precaution}\n")
        self.output_text.configure(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
