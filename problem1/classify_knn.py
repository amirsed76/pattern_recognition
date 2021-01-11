from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from memory_profiler import profile


@profile
def knn_classifier(k, x_train, y_train, x_test) -> pd.DataFrame:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    return prediction


def classify_toy_data():
    info = {}
    for problem in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        df_train = pd.read_csv(filepath_or_buffer=f"data/{problem}_train.csv")
        df_test = pd.read_csv(filepath_or_buffer=f"data/{problem}_test.csv")
        prediction = knn_classifier(k=10, x_train=df_train[["x1", "x2"]], y_train=df_train["label"], x_test=df_test)
        true_label = list(np.full(shape=int(len(prediction) / 2), fill_value=1, dtype=np.int))
        true_label.extend(
            list(np.full(shape=(int(len(prediction)) - int(len(prediction) / 2)), fill_value=2, dtype=np.int)))
        # result["truth_label"] = truth_label
        result = pd.DataFrame({
            "prediction": prediction,
            "true_label": true_label
        })
        result.to_csv(f"result/predictions/{problem}.csv", index=False)

        info[problem] = {
            "confusion_matrix": str(confusion_matrix(y_true=result["true_label"], y_pred=result["prediction"])),
            "accuracy": accuracy_score(y_true=result["true_label"], y_pred=result["prediction"])
        }

    with open("result/problem1_1.json", "w") as file:
        file.write(str(info))


if __name__ == '__main__':
    classify_toy_data()
