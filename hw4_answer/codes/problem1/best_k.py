from problem1.classify_knn import knn_classifier
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


def plot_k_accuracy(x_train_data, y_train_data, validation_data_x, validation_data_y, path):
    accuracies = []
    for k in range(1, 200):
        predictions = knn_classifier(k=k, x_train=x_train_data, y_train=y_train_data, x_test=validation_data_x)
        accuracies.append(accuracy_score(y_true=validation_data_y, y_pred=predictions))

    plt.plot(range(1, 200), accuracies)
    plt.xlabel(f" k (best k = {accuracies.index(max(accuracies)) + 1})")
    plt.ylabel("accuracy")
    plt.savefig(path)
    plt.close()

    return accuracies.index(max(accuracies)) + 1


if __name__ == '__main__':
    for problem in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        df_train = pd.read_csv(filepath_or_buffer=f"data/{problem}_train.csv")
        x_data1 = df_train.iloc[0:900, :]
        validation_data1 = df_train.iloc[900:1000, :]
        x_data2 = df_train.iloc[1000:1900, :]
        validation_data2 = df_train.iloc[1900:2000, :]

        data = pd.concat([x_data1, x_data2])
        validation_data = pd.concat([validation_data1, validation_data2])
        best_k = plot_k_accuracy(x_train_data=data[["x1", "x2"]], y_train_data=data["label"],
                                 validation_data_x=validation_data[["x1", "x2"]],
                                 validation_data_y=validation_data["label"],
                                 path=f"result/k_accuracy_plots/{problem}.png")
