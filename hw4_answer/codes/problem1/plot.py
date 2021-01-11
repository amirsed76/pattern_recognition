from matplotlib import pyplot as plt
import pandas as pd


def draw_plot(df: pd.DataFrame, path):
    plt.scatter(x=df["x1"] , y=df["x2"] , c=df["label"])
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    for problem in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        df_train = pd.read_csv(filepath_or_buffer=f"data/{problem}_train.csv")
        draw_plot(df_train , path =f"result/plots/{problem}.png")
