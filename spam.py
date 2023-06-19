import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = None
classifier = None
data = None

def load_data():
    global data
    file_path = filedialog.askopenfilename(title="Select data file", filetypes=(("CSV files", "*.csv"),))
    if file_path:
        try:
            data = pd.read_csv(file_path)
            messagebox.showinfo("Data Loaded", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", "Error loading data: {}".format(str(e)))
    else:
        messagebox.showwarning("No File Selected", "No data file selected.")

def train_model():
    global vectorizer, classifier, data
    if data is not None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['EmailText'])
        y = data['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

        accuracy = classifier.score(X_test, y_test)
        messagebox.showinfo("Training Complete", "Model trained successfully!\nAccuracy: {:.2f}".format(accuracy))
    else:
        messagebox.showwarning("No Data", "Please load the data before training the model.")

def detect_spam(email_text):
    global vectorizer, classifier
    if vectorizer is not None and classifier is not None:
        email_text = vectorizer.transform([email_text])
        prediction = classifier.predict(email_text)[0]
        messagebox.showinfo("Spam Detection Result", "The email is {}".format('spam' if prediction == 'spam' else 'not spam'))
    else:
        messagebox.showwarning("Model Not Trained", "Please train the model before detecting spam.")

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select email file", filetypes=(("Text files", "*.txt"),))
    if file_path:
        try:
            with open(file_path, 'r') as file:
                email_text = file.read()
                detect_spam(email_text)
                email_text_box.delete(1.0, tk.END)
                email_text_box.insert(tk.END, email_text)
        except Exception as e:
            messagebox.showerror("Error", "Error opening file: {}".format(str(e)))
    else:
        messagebox.showwarning("No File Selected", "No email file selected.")

# UI Setup
root = tk.Tk()
root.title("Spam Email Detector")
root.geometry("600x400")
root.config(bg="#ECECEC")

# Frames
top_frame = tk.Frame(root, bg="#ECECEC")
top_frame.pack(pady=20)

bottom_frame = tk.Frame(root, bg="#ECECEC")
bottom_frame.pack(pady=10)

# Labels
title_label = tk.Label(top_frame, text="Spam Email Detector", font=("Arial", 24), bg="#ECECEC")
title_label.pack()

email_label = tk.Label(bottom_frame, text="Email Text:", font=("Arial", 16), bg="#ECECEC")
email_label.pack(side=tk.LEFT)

# Buttons
load_data_button = tk.Button(top_frame, text="Load Data", command=load_data, bg="#0078D4", fg="white", font=("Arial", 14))
load_data_button.pack(padx=10, side=tk.LEFT)

train_model_button = tk.Button(top_frame, text="Train Model", command=train_model, bg="#0078D4", fg="white", font=("Arial", 14))
train_model_button.pack(padx=10, side=tk.LEFT)

detect_spam_button = tk.Button(top_frame, text="Detect Spam", command=open_file_dialog, bg="#0078D4", fg="white", font=("Arial", 14))
detect_spam_button.pack(padx=10, side=tk.LEFT)

# Text box
email_text_box = tk.Text(bottom_frame, width=60, height=15)
email_text_box.pack(side=tk.RIGHT)

root.mainloop()
