from decimal import Decimal
import pandas as pd
import numpy as np

if __name__ == '__main__':

    result = {}
    KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    with open("toy_data.txt") as file:
        contents = file.read().split("******\n")
        for index, content in enumerate(contents):
            test, train1, train2 = content.split("____")
            tests = test.strip().split("\n")
            test = pd.DataFrame({
                "x1": [Decimal(x.strip()) for x in tests[0].replace("\x00", " ").split(" ") if x.strip() != ""],
                "x2": [Decimal(x.strip()) for x in tests[1].replace("\x00", " ").split(" ") if x.strip() != ""]
            })

            trains1 = train1.strip().split("\n")
            train1 = pd.DataFrame({
                "x1": [Decimal(x.strip()) for x in trains1[0].replace("\x00", " ").split(" ") if x.strip() != ""],
                "x2": [Decimal(x.strip()) for x in trains1[1].replace("\x00", " ").split(" ") if x.strip() != ""],
                "label": np.full(shape=len(trains1[0].replace("\x00", " ").split(" ")), fill_value=1, dtype=np.int)
            })

            trains2 = train2.strip().split("\n")
            train2 = pd.DataFrame({
                "x1": [Decimal(x.strip()) for x in trains2[0].replace("\x00", " ").split(" ") if x.strip() != ""],
                "x2": [Decimal(x.strip()) for x in trains2[1].replace("\x00", " ").split(" ") if x.strip() != ""],
                "label": np.full(shape=len(trains2[0].replace("\x00", " ").split(" ")), fill_value=2, dtype=np.int)
            })
            train = pd.concat([train1, train2])

            pd.DataFrame(test).to_csv(f"data/{KEYS[index]}_test.csv", index=False)
            pd.DataFrame(train).to_csv(f"data/{KEYS[index]}_train.csv", index=False)
