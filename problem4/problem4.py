from sklearn.naive_bayes import GaussianNB
from matplotlib.image import imread
import numpy as np
import pandas
from sklearn.metrics import confusion_matrix, accuracy_score

from problem1.classify_knn import knn_classifier


def load_images(path):
    img = imread(path)
    images = []
    for i in range(34):
        for j in range(33):
            m = img[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
            if not (j == 32 and i >= 12):
                images.append(np.squeeze(m.T.reshape(-1)))
    return images


def get_data():
    trains = pandas.DataFrame()
    tests = pandas.DataFrame()
    for i in range(1, 6):
        image_data = load_images(f"images/usps_{i}.jpg")
        df = pandas.DataFrame(image_data)
        train_df = pandas.DataFrame(image_data[0:int(0.5 * len(image_data))])
        train_df["label"] = i
        test_df = pandas.DataFrame(image_data[int(0.5 * len(image_data)):])
        test_df["label"] = i

        trains = pandas.concat([trains.copy(), train_df])
        tests = pandas.concat([tests.copy(), test_df])
    return trains, tests


if __name__ == '__main__':
    trains, tests = get_data()
    model = GaussianNB()
    model.fit(trains.iloc[:, :-1], trains["label"])
    prediction = model.predict(tests.iloc[:, :-1])
    result_df = pandas.DataFrame({
        "true": tests["label"],
        "prediction": prediction
    })
    info = {
        "confusion_matrix": str(confusion_matrix(y_true=tests["label"], y_pred=prediction)),
        "accuracy": accuracy_score(y_true=tests["label"], y_pred=prediction)
    }
    result_df.to_csv(path_or_buf="results/prediction.csv", index=False)
    with open("results/accuracy.json", "w") as f:
        f.write(str(info))
