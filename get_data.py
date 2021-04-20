import numpy as np
import re
import os.path
from sklearn.datasets import make_multilabel_classification


def create_dataset():
    dates = []
    numbers = []
    h4_rows = []
    class_id = "numbers text-nowrap"
    start = '<h4>'
    end = '</h4>'
    stake = "le\">(\d+)</span>"

    # my_file = "04_2021_2.mhtml"

    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    orders = ["1", "2", "3", "4"]

    my_files = []
    for year in years:
        for month in months:
            for order in orders:
                my_file = "P:/WORK/high5/" + year + "_" + month + "_" + order + ".mhtml"
                if os.path.isfile(my_file):
                    my_files.append(my_file)

    print("\nnumber of files: ", len(my_files))

    for k in range(len(my_files)):
        # must make sure that files are in ordered
        my_file = my_files[k]

        with open(my_file, "r") as file:
            lines = file.readlines()
            for line in lines:

                # get lottery number
                number = re.search(stake, line)
                if number:
                    numbers.append(number.group(1))

                # get date
                date = re.search('%s(.*)%s' % (start, end), line)
                if date:
                    date = date.group(1)
                    for year in years:
                        if year in date:
                            # replace strange string
                            date = date.replace("F=C3=A9vrier", "Février")
                            date = date.replace("Ao=C3=BBt", "Août")
                            date = date.replace("D=C3=A9cembre", "Décembre")
                            dates.append(date)

    assert len(numbers) % 5 == 0
    numbers = np.array(numbers).reshape((int(len(numbers)/5), 5)).astype('int16')
    assert len(numbers) == len(dates)

    print("\nnumber of days: ", len(dates))

    np.savetxt("dates.txt", np.array(dates), fmt='%s')
    np.savetxt("numbers.txt", numbers)

    # for i in range(len(dates)):
    #     print(dates[i], "\t\t", numbers[i])


def get_data(look_back=1, look_next=1, train_ratio=0.8, is_reshape=False):
    # dates = np.loadtxt("dates.txt")
    numbers = np.loadtxt("numbers.txt")

    dataset = np.zeros(shape=(len(numbers), 32))
    for i in range(len(numbers)):
        for j in range(len(numbers[i])):
            index = int(numbers[i, j]) - 1  # 0 to 31 equivalent to 1-32
            dataset[i, index] = 1.

    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back - look_next, look_next):
        dataX.append(dataset[i:(i + look_back), :])
        # dataY.append(dataset[(i + look_back):(i + look_back + look_next), :])
        dataY.append(dataset[(i + look_back), :])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if is_reshape:
        dataX = np.reshape(dataX, (dataX.shape[0], 32, dataX.shape[1]))
        # dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 32))
    train_size = int(len(dataY) * train_ratio)
    trainX = dataX[:train_size]
    trainY = dataY[:train_size]
    testX = dataX[train_size:]
    testY = dataY[train_size:]

    return trainX, trainY, testX, testY

