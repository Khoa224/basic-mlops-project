import pandas as pd
from sklearn.datasets import make_classification
import os


# create sample dataset
def create_data():
    if not os.path.exists("data"):
        os.mkdir("data")

    append_mode = os.path.isfile("data/train.csv")

    num_dataset = 10 if not append_mode else 1

    for _ in range (num_dataset):
        X, y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, n_classes=2, random_state=42)

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range (X.shape[1])])
        df['target'] = y

        train_data = df.iloc[:8000]
        test_data = df.iloc[8000:]

        train_data.to_csv("data/train.csv", mode="a", header=not append_mode, index=False)
        test_data.to_csv("data/test.csv", mode="a", header=not append_mode, index=False)

        print("Data extracted successfully")


if __name__ == "__main__":
    create_data()