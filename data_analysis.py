#!/usr/bin/python
import copy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def clean_realtime_data(file_name):
    """
    Clean the data exported directly from the CWW database.
    :param file_name: real-time file
    :return:
    """
    # read the data from csv file to list
    all_datetime = []
    measurements = []
    fr = open(file_name, "r")
    fr.readline()
    for line in fr.readlines():
        m = line.strip().split(",")
        cur_datetime = datetime.datetime.strptime(m[0][:19], "%Y-%m-%d %H:%M:%S")
        all_datetime.append(cur_datetime)  # all_datetime
        measurements.append([cur_datetime, m[1], m[2], m[3], m[5], m[6]])  # measurements
    fr.close()

    # set non-number entries to nan
    for m in measurements:
        for i in range(1, 6):
            if m[i].startswith("["):  # replace the data with [*] with nan
                m[i] = str(float("nan"))
            if m[i] == "NULL":  # replace the data with NULL with nan
                m[i] = str(float("nan"))

    # deal with missing and duplicated measurement
    clean_measurements = []
    min_datetime = min(all_datetime)
    max_datetime = max(all_datetime)
    cur_ts = min_datetime
    while cur_ts <= max_datetime:
        if cur_ts in all_datetime:
            idx = all_datetime.index(cur_ts)
            cm = copy.deepcopy(measurements[idx])
            cm[0] = str(cm[0])
            cm.append("0")
            clean_measurements.append(cm)
        else:
            cm = [str(cur_ts), str(float("nan")), str(float("nan")),
                  str(float("nan")), str(float("nan")), str(float("nan")), "0"]
            clean_measurements.append(cm)
        cur_ts += datetime.timedelta(hours=1)

    # write the cleaned data from list to file

    output_file = "ardec_cleaned.csv"
    fw = open(output_file, "w")
    fw.write("MeasurementDateTime,Temperature,pH,Conductivity,ORP,DO,Label\n")
    for cm in clean_measurements:
        line = ",".join(cm) + "\n"
        fw.write(line)
    fw.close()
    #
    print "clean data finished!"


def plot_features_trend(chill_file, from_datetime, to_datetime):
    """
    Plot the trend of each feature
    :param chill_file: chill_cleaned
    :return:
    """
    # convert datetime string to datetime object
    from_datetime = datetime.datetime.strptime(from_datetime, "%Y-%m-%d %H:%M:%S")
    to_datetime = datetime.datetime.strptime(to_datetime, "%Y-%m-%d %H:%M:%S")

    # read data from file to list
    fr = open(chill_file, "r")
    fr.readline()
    my_datetime = []
    my_values = []
    fr = open(chill_file, "r")
    fr.readline()
    for line in fr.readlines():
        m = line.strip().split(",")
        cur_datetime = datetime.datetime.strptime(m[0][:19], "%Y-%m-%d %H:%M:%S")
        if from_datetime <= cur_datetime <= to_datetime and m[6] == "1":
            my_datetime.append(cur_datetime)  # hist_datetime
            my_values.append([float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5]), float(m[6])])
    fr.close()
    my_values = np.array(my_values, dtype=float)
    # plot
    print my_values
    temp = my_values[:, 0]
    ph = my_values[:, 1]
    cond = my_values[:, 2]
    orp = my_values[:, 3]
    do = my_values[:, 4]
    #
    fig = plt.figure()
    fig.suptitle("All Data")
    #
    ax_temp = fig.add_subplot(511)
    ax_temp.plot(my_datetime, temp, "-r")
    ax_temp.set_ylabel("Temperature")
    #
    ax_ph = fig.add_subplot(512)
    ax_ph.plot(my_datetime, ph, "-g")
    ax_ph.set_ylabel("pH")
    #
    ax_cond = fig.add_subplot(513)
    ax_cond.plot(my_datetime, cond, "-b")
    ax_cond.set_ylabel("Conductivity")
    #
    ax_orp = fig.add_subplot(514)
    ax_orp.plot(my_datetime, orp, "-m")
    ax_orp.set_ylabel("ORP")
    #
    ax_do = fig.add_subplot(515)
    ax_do.plot(my_datetime, do, "-c")
    ax_do.set_ylabel("DO")
    ax_do.set_xlabel("Date Time")
    #
    plt.show()


def plot_features_histogram(chill_file, from_datetime, to_datetime):
    """
    Plot the histogram graph for each feature to check whether it's normal distribution.
    :param chill_file: chill_labeled
    :param from_datetime: datetime string as "2014-2-4 1:00:00"
    :param to_datetime: datetime string as "2015-2-4 00:00:00"
    :return:
    """
    # convert datetime string to datetime object
    from_datetime = datetime.datetime.strptime(from_datetime, "%Y-%m-%d %H:%M:%S")
    to_datetime = datetime.datetime.strptime(to_datetime, "%Y-%m-%d %H:%M:%S")

    # load the historical data
    hist_data = []
    fr = open(chill_file, "r")
    fr.readline()
    for line in fr.readlines():
        m = line.strip().split(",")
        cur_datetime = datetime.datetime.strptime(m[0], "%Y-%m-%d %H:%M:%S")
        if from_datetime <= cur_datetime <= to_datetime and m[6] == "1":
                hist_data.append([float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])])
    fr.close()
    hist_data = np.array(hist_data, dtype=float)

    # plot the histogram of each feature
    plt.hist(hist_data[:, 0], color="r")
    plt.title("Temperature")
    plt.show()
    #
    plt.hist(hist_data[:, 1], color="g")
    plt.title("pH")
    plt.show()
    #
    plt.hist(hist_data[:, 2], color="b")
    plt.title("Conductivity")
    plt.show()
    #
    plt.hist(hist_data[:, 3], color="m")
    plt.title("ORP")
    plt.show()
    #
    plt.hist(hist_data[:, 4], color="c")
    plt.title("DO")
    plt.show()


def label_historical_data(chill_file, from_datetime, to_datetime):
    """
    Label each historical measurement based on the theory of Chebyshev inequality.
    :param chill_file: the cleaned csv file
    :param from_datetime: datetime string as "2014-2-4 1:00:00"
    :param to_datetime: datetime string as "2015-2-4 00:00:00"
    :return:
    """
    # convert datetime string to datetime object
    from_datetime = datetime.datetime.strptime(from_datetime, "%Y-%m-%d %H:%M:%S")
    to_datetime = datetime.datetime.strptime(to_datetime, "%Y-%m-%d %H:%M:%S")

    # read data into list measured from_datetime to to_datetime
    hist_datetime = []
    hist_values = []  #
    fr = open(chill_file, "r")
    fr.readline()
    for line in fr.readlines():
        m = line.strip().split(",")
        cur_datetime = datetime.datetime.strptime(m[0][:19], "%Y-%m-%d %H:%M:%S")
        if from_datetime <= cur_datetime <= to_datetime:
            hist_datetime.append(cur_datetime)  # hist_datetime
            hist_values.append([m[1], m[2], m[3], m[4], m[5], m[6]])  # hist_values
    fr.close()

    # label -1 to the measurement with nan as outlier
    for i in range(len(hist_values)):
        for j in range(5):
            if hist_values[i][j] == "nan":
                hist_values[i][-1] = "-1"  # -1 means outlier

    # calculate the mean and standard deviation for each feature without nan
    hist_values = np.array(hist_values, dtype=float)
    mu = np.nanmean(hist_values, axis=0)[:5]
    sigma = np.nanstd(hist_values, axis=0)[:5]

    # Chebyshev inequality, reference https://en.wikipedia.org/wiki/Chebyshev's_inequality
    # k*sigma, k=3 88.8889%, k=4 93.75%, k=5 96%
    for i in range(len(hist_values)):
        outlier_flag = 1
        for j in range(5):
            if not math.isnan(hist_values[i][j]):
                k = (np.abs(hist_values[i][j] - mu[j])) / sigma[j]
                if k >= 2.56:
                    outlier_flag = -1
        if not math.isnan(hist_values[i][j]):
            hist_values[i][-1] = outlier_flag
            if outlier_flag == -1:
                print hist_values[i]

    # write labeled result into file
    fw = open("ardec_labeled.csv", "w")
    fw.write("MeasurementDateTime,Temperature,pH,Conductivity,ORP,DO,Label\n")
    fr = open(chill_file, "r")
    fr.readline()
    lines = fr.readlines()
    for i in range(len(hist_datetime)):
        m = lines[i].strip().split(",")
        m[-1] = str(int(hist_values[i][-1]))
        line = ",".join(m) + "\n"
        fw.write(line)
    fr.close()
    fw.close()
    #
    print "historical data labeled!"


def load_data(chill_file, from_datetime, to_datetime, label=0):
    """
    Load trained data with label 1 in the period from the from_datetime to to_datetime.
    :param chill_file: chill_labeled.csv
    :param from_datetime: datetime string as "2014-2-4 1:00:00"
    :param to_datetime: datetime string as "2015-2-4 00:00:00"
    :param type: 0, 1 and -1
    :return:
    """
    # convert datetime string to datetime object
    from_datetime = datetime.datetime.strptime(from_datetime, "%Y-%m-%d %H:%M:%S")
    to_datetime = datetime.datetime.strptime(to_datetime, "%Y-%m-%d %H:%M:%S")

    # load train data
    fr = open(chill_file, "r")
    fr.readline()
    date_time = []
    values = []
    labels = []
    for line in fr.readlines():
        m = line.strip().split(",")
        cur_datetime = datetime.datetime.strptime(m[0], "%Y-%m-%d %H:%M:%S")
        if label == 1:
            if from_datetime <= cur_datetime <= to_datetime and m[6] == "1":
                date_time.append(cur_datetime)
                values.append([float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])])
                labels.append(float(m[6]))
        if label == 0:
            if from_datetime <= cur_datetime <= to_datetime:
                date_time.append(cur_datetime)
                values.append([float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])])
                labels.append(float(m[6]))
        if label == -1:
            if from_datetime <= cur_datetime <= to_datetime and m[6] == "-1":
                date_time.append(cur_datetime)
                values.append([float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])])
                labels.append(float(m[6]))
    fr.close()
    values = np.array(values, dtype=float)
    return values, date_time, labels


def detect_outlier(data_train, measurement):
    """
    Detect whether the input measurement is outlier or not.
    :param data_train: data for training the one class SVM model
    :param measurement: one row from the chill_untested.csv
    :return: predicted label for input measurement
    """
    classifier = svm.OneClassSVM(kernel="rbf", nu=0.005, gamma=0.00001)
    classifier.fit(data_train)
    label = classifier.predict(measurement)[0]
    return label


def start_monitoring(chill_labeled, chill_untested):
    """
    Start the monitoring work in an online way
    :param chill_labeled: for loading train data
    :param chill_untested: for read untested measurements
    :return:
    """
    fr = open(chill_untested, "r")
    fr.readline()
    lines = fr.readlines()
    fr.close()
    for line in lines:
        # new measurement
        m = line.strip().split(",")
        cur_datetime = datetime.datetime.strptime(m[0], "%Y-%m-%d %H:%M:%S")

        # detect outlier
        nan_flag = 0
        for j in range(1, 6):
            if m[j] == "nan":
                nan_flag = 1
        if nan_flag:
            label = "-1"
            # update historical data
            new_line = ",".join([str(cur_datetime), m[1], m[2], m[3], m[4], m[5], label]) + "\n"
            fw = open(chill_labeled, "a+")
            fw.write(new_line)
            fw.close()
            print new_line.strip()
        else:
            data_train = load_data(chill_labeled, "2014-2-13 00:00:00", "2015-2-12 23:00:00", 1)[0]
            cur_measurement = [float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])]
            label = detect_outlier(data_train, cur_measurement)
            # update historical data
            new_line = ",".join([str(cur_datetime), m[1], m[2], m[3], m[4], m[5], str(int(label))]) + "\n"
            fw = open(chill_labeled, "a+")
            fw.write(new_line)
            fw.close()
            print new_line.strip()


def visualize_outliers(chill_file, from_datetime, to_datetime):
    """
    Visualize the the outlier
    :param chill_file: chill_labeled
    :param from_datetime: datetime string as "2014-2-4 1:00:00"
    :param to_datetime: datetime string as "2015-2-4 00:00:00"
    :return:
    """
    values, date_time, labels = load_data(chill_file, from_datetime, to_datetime, 0)
    fig = plt.figure()
    fig.suptitle("Outliers")
    #
    ax_temp = fig.add_subplot(611)
    ax_temp.plot(date_time, values[:, 0], "-r")
    ax_temp.set_ylabel("Temperature")
    #
    ax_ph = fig.add_subplot(612)
    ax_ph.plot(date_time, values[:, 1], "-g")
    ax_ph.set_ylabel("pH")
    #
    ax_cond = fig.add_subplot(613)
    ax_cond.plot(date_time, values[:, 2], "-b")
    ax_cond.set_ylabel("Conductivity")
    #
    ax_orp = fig.add_subplot(614)
    ax_orp.plot(date_time, values[:, 3], "-m")
    ax_orp.set_ylabel("ORP")
    #
    ax_do = fig.add_subplot(615)
    ax_do.plot(date_time, values[:, 4], "-c")
    ax_do.set_ylabel("DO")
    #
    ax_lb = fig.add_subplot(616)
    ax_lb.plot(date_time, labels, "-y")
    ax_lb.set_ylim([-2, 2])
    ax_lb.set_yticks([-1, 1])
    ax_lb.set_ylabel("Label")
    ax_lb.set_xlabel("Date Time")
    #
    plt.show()


if __name__ == "__main__":
    # clean_realtime_data("chill_realtime.csv")
    # plot_features_trend("chill_labeled.csv", "2014-2-4 01:00:00", "2015-2-4 00:00:00")
    # label_historical_data("chill_cleaned.csv", "2014-2-4 01:00:00", "2015-2-4 00:00:00")
    # start_monitoring("chill_labeled.csv", "chill_untested.csv")
    # visualize_outliers("chill_labeled_0006.csv", "2015-2-4 01:00:00", "2015-12-3 06:00:00")

    # clean_realtime_data("ardec_realtime.csv")
    # plot_features_trend("chill_labeled.csv", "2014-2-4 01:00:00", "2015-2-4 00:00:00")
    label_historical_data("ardec_cleaned.csv", "2014-2-13 00:00:00", "2015-2-12 23:00:00")
    # start_monitoring("ardec_labeled.csv", "ardec_untested.csv")
    # visualize_outliers("ardec_labeled.csv", "2015-2-13 01:00:00", "2015-12-3 06:00:00")
