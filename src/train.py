import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # Allows us to pass values from the terminal
    parser = argparse.ArgumentParser(description="Train an Iris classifier")

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("Feature names:", iris.feature_names)
    print("Target names:", iris.target_names)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Train the model
    model = DecisionTreeClassifier(max_depth=4, random_state=args.random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    print("Predictions:", y_pred)
    print("Actual:", y_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)

    # Create outputs folder automatically
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Save confusion matrix as a PNG image
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Iris Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outputs_dir / "confusion_matrix.png")
    plt.close()

    # Save trained model
    joblib.dump(model, outputs_dir / "model.joblib")

    print("Saved confusion matrix to outputs/confusion_matrix.png")
    print("Saved model to outputs/model.joblib")


if __name__ == "__main__":
    main()