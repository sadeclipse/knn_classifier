import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report

iris_dataset = load_iris()
df = pd.DataFrame(iris_dataset["data"], columns=iris_dataset.feature_names)
df["target"] = iris_dataset["target"]
print(df)


def split_df_train(df: pd.DataFrame, ratio=0.2):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    split_index = int(len(df_shuffled) * (1 - ratio))
    df_train = df_shuffled.iloc[:split_index]
    df_test = df_shuffled.iloc[split_index:]
    x_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    x_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]
    return x_train, y_train, x_test, y_test


class knn_classifier:
    def __init__(self, k: int = 1):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train.values
        self.y_train = y_train.values
        return self

    def calc_distance(self, point1: np.ndarray, point2: np.ndarray):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def calc_majority(self, targets):
        return max(set(targets), key=targets.count)

    def calculate_k_nearest(self, test_point):
        distances = [
            self.calc_distance(test_point, train_point) for train_point in self.x_train
        ]
        k_nearest_ind = np.argsort(distances)[: self.k]
        k_nearest_targets = [self.y_train[index] for index in k_nearest_ind]
        return k_nearest_targets

    def predict(self, x: pd.DataFrame):
        predictions = []
        x = x.values

        for test_point in x:
            targets = self.calculate_k_nearest(test_point=test_point)
            target_prediction = self.calc_majority(targets=targets)
            predictions.append(target_prediction)
        return predictions


x_train, y_train, x_test, y_test = split_df_train(df=df)
knn = knn_classifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = np.mean(y_pred == y_test)


def add_predictions_and_compare(x_test, y_test, y_pred, target_col_name="target"):
    """
    Creates a DataFrame with features, true labels, and predictions side-by-side.

    Parameters:
    - x_test: DataFrame or array of test features
    - y_test: true labels (array-like)
    - y_pred: predicted labels (array-like)
    - target_col_name: name for the true label column (default: "target")

    Returns:
    - DataFrame with columns: [feature1, feature2, ..., target, predicted]
    """
    # Convert inputs to consistent format
    x_test_df = pd.DataFrame(x_test).copy()
    y_test = pd.Series(y_test, name=target_col_name, index=x_test_df.index)
    y_pred = pd.Series(y_pred, name="predicted", index=x_test_df.index)

    # Combine into one DataFrame
    result_df = pd.concat([x_test_df, y_test, y_pred], axis=1)

    # Highlight mismatches (optional but useful)
    result_df["correct"] = result_df[target_col_name] == result_df["predicted"]

    return result_df


comparison = add_predictions_and_compare(
    x_test=x_test, y_test=y_test, y_pred=y_pred, target_col_name="target"
)

# Show all rows (or just the mistakes)
print("\n=== Full Comparison ===")
print(comparison)

print("\n=== Only Mistakes ===")
print(comparison[~comparison["correct"]])

print(f"\n✅ Overall Accuracy: {comparison['correct'].mean():.2%}")
