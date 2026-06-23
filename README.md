Iris Classifier Decision Tree
Overview

This project is a simple machine learning example using the classic Iris dataset from scikit-learn.
The aim of the project is to train a Decision Tree classifier to predict the species of an Iris flower based on flower measurements.

The project demonstrates a basic machine learning workflow:

Loading a dataset
Splitting data into training and testing sets
Training a model
Making predictions
Evaluating the model using accuracy and a confusion matrix
Saving the trained model and confusion matrix output
Project Structure
iris-classifier/
├── data/
├── notebooks/
│   └── iris_model.ipynb
├── outputs/
│   ├── confusion_matrix.png
│   └── model.joblib
├── src/
│   └── train.py
├── tests/
│   └── test_train.py
├── .gitignore
├── license.txt
├── readme.md
└── requirements.txt
Quick Start

Clone the repository:

git clone YOUR_GITHUB_REPO_LINK_HERE
cd iris-classifier

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt

Run the training script:

python src/train.py --test-size 0.2 --random-state 42
Outputs

After running the script, the following files are created inside the outputs/ folder:

confusion_matrix.png
model.joblib

confusion_matrix.png shows the model performance visually.

model.joblib saves the trained machine learning model so it can be loaded again later.

Model

The model used in this project is a Decision Tree Classifier from scikit-learn.

Dataset

The Iris dataset is loaded directly from scikit-learn, so no external dataset file is needed.

Requirements

The required Python packages are listed in requirements.txt.

License

This project uses the MIT License
