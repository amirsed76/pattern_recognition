import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
import pandas as pd
from matplotlib import pyplot as plt


def class_multivariate_normal(mu, cov_matrix):
    N = multivariate_normal(mu, cov_matrix)
    return N


def bayes_classifier(priors, test_data, classes_info):
    classes_N = []
    for class_info in classes_info:
        classes_N.append(class_multivariate_normal(mu=class_info["mu"], cov_matrix=class_info["cov_matrix"]))

    predictions = []
    for x in test_data:
        posteriors = np.array([classes_N[i].pdf(x) * priors[i] for i in range(len(priors))])  # p(x|wi)
        predictions.append(posteriors.argmax() + 1)
    return predictions


def problem2_1():
    problems = [
        {
            "name": "A",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, -0.8],
                                   [-0.8, 1]]
                },
                {
                    "mu": [2, 2],
                    "cov_matrix": [[1, 0.8],
                                   [0.8, 1]]
                }
            ]
        },
        {
            "name": "B",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, -0.75],
                                   [-0.75, 1]]
                },
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, 0.75],
                                   [0.75, 1]]
                }
            ]
        },
        {
            "name": "C",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1.75, 0],
                                   [0, 0.25]]
                },
                {
                    "mu": [0, 0],
                    "cov_matrix": [[0.25, 0],
                                   [0, 1.75]]
                }
            ]
        },
        {
            "name": "D",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, 0],
                                   [0, 1]]
                },
                {
                    "mu": [0, 0],
                    "cov_matrix": [[9, 0],
                                   [0, 9]]
                }
            ]
        },
        {
            "name": "E",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[3, 1],
                                   [1, 0.5]]
                },
                {
                    "mu": [-0.8, 0.8],
                    "cov_matrix": [[3, 1],
                                   [1, 0.5]]
                }
            ]
        },
        {
            "name": "F",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[3, 1],
                                   [1, 0.5]]
                },
                {
                    "mu": [1, 6],
                    "cov_matrix": [[3, 1],
                                   [1, 0.5]]
                }
            ]
        }

    ]
    info = {}
    for problem_info in problems:
        problem = problem_info["name"]
        test_data = pd.read_csv(filepath_or_buffer=f"data/{problem}_test.csv").values.tolist()
        prediction = bayes_classifier(priors=[0.5, 0.5], test_data=test_data, classes_info=problem_info["classes"])

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

    with open("result/problem2_1.json", "w") as file:
        file.write(str(info))


def problem2_2():
    problems = [
        {
            "name": "A",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, 0],
                                   [0, 1]]
                },
                {
                    "mu": [2, 2],
                    "cov_matrix": [[1, 0],
                                   [0, 1]]
                }
            ]
        },
        {
            "name": "B",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, 0],
                                   [0, 1]]
                },
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, 0],
                                   [0, 1]]
                }
            ]
        },
        {
            "name": "C",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1.75, 0],
                                   [0, 0.25]]
                },
                {
                    "mu": [0, 0],
                    "cov_matrix": [[0.25, 0],
                                   [0, 1.75]]
                }
            ]
        },
        {
            "name": "D",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[1, 0],
                                   [0, 1]]
                },
                {
                    "mu": [0, 0],
                    "cov_matrix": [[9, 0],
                                   [0, 9]]
                }
            ]
        },
        {
            "name": "E",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[3, 0],
                                   [0, 0.5]]
                },
                {
                    "mu": [-0.8, 0.8],
                    "cov_matrix": [[3, 0],
                                   [0, 0.5]]
                }
            ]
        },
        {
            "name": "F",
            "classes": [
                {
                    "mu": [0, 0],
                    "cov_matrix": [[3, 0],
                                   [0, 0.5]]
                },
                {
                    "mu": [1, 6],
                    "cov_matrix": [[3, 0],
                                   [0, 0.5]]
                }
            ]
        }

    ]
    info = {}
    for problem_info in problems:
        problem = problem_info["name"]
        test_data = pd.read_csv(filepath_or_buffer=f"data/{problem}_test.csv").values.tolist()
        prediction = bayes_classifier(priors=[0.5, 0.5], test_data=test_data, classes_info=problem_info["classes"])

        true_label = list(np.full(shape=int(len(prediction) / 2), fill_value=1, dtype=np.int))
        true_label.extend(
            list(np.full(shape=(int(len(prediction)) - int(len(prediction) / 2)), fill_value=2, dtype=np.int)))
        # result["truth_label"] = truth_label
        result = pd.DataFrame({
            "prediction": prediction,
            "true_label": true_label
        })

        info[problem] = {
            "confusion_matrix": str(confusion_matrix(y_true=result["true_label"], y_pred=result["prediction"])),
            "accuracy": accuracy_score(y_true=result["true_label"], y_pred=result["prediction"])
        }

    with open("result/problem2_2.json", "w") as file:
        file.write(str(info))


if __name__ == '__main__':
    problem2_1()
    problem2_2()
