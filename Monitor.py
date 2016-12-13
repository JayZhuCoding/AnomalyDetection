#!/usr/bin/python
import copy
import datetime
from decimal import Decimal
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn import svm


class Monitor:

    def __init__(self, name, input_file):
        """
        :param name: give monitor a name
        :param input_file: the csv file with real-time data
        :return:
        """
        self.name = name
        self.input_file = input_file
        self.monitor_file = None
        self.min_timestamp = None
        self.max_timestamp = None

    def initialize_monitor(self):
        timestamps = []
        fr = open(self.input_file, "r")  # read real-time data into list
        fr.readline()
        for line in fr.readlines():
            m = line.strip().split(",")
            timestamp = datetime.datetime.strptime(m[0][:13], "%Y-%m-%d %H")
            timestamps.append(timestamp)  # timestamps
        fr.close()
        self.min_timestamp = min(timestamps)
        self.max_timestamp = max(timestamps)
        self.monitor_file = os.path.join(os.path.dirname(self.input_file), self.name + "_Monitor.csv")

    def create_monitor_file(self):
        timestamps = []
        measurements = []
        fr = open(self.input_file, "r")  # read real-time data into list
        fr.readline()
        for line in fr.readlines():
            m = line.strip().split(",")
            timestamp = datetime.datetime.strptime(m[0][:13], "%Y-%m-%d %H")
            temperature = m[1]
            ph = m[2]
            conductivity = m[3]
            orp = m[5]
            do = m[6]
            timestamps.append(timestamp)  # timestamps
            measurements.append([timestamp, temperature, ph, conductivity, orp, do])  # measurements
        fr.close()
        for m in measurements:
            for i in range(1, 6):
                if m[i].startswith("["):  # replace the data with [*] with nan
                    m[i] = "nan"
                if m[i] == "NULL":  # replace the data with NULL with nan
                    m[i] = "nan"
        clean_measurements = []
        unique_timestamp = self.min_timestamp
        while unique_timestamp <= self.max_timestamp:  # deal with missing and duplicated measurement
            if unique_timestamp in timestamps:
                cm = measurements[timestamps.index(unique_timestamp)]
                timestamp = str(unique_timestamp)
                temperature = cm[1]
                ph = cm[2]
                conductivity = cm[3]
                orp = cm[4]
                do = cm[5]
                label = "nan"
                outlier_prob = "nan"
                if temperature == "nan" or ph == "nan" or conductivity == "nan" or orp == "nan" or do == "nan":
                    label = "-1"
                    outlier_prob = "1.0"
                event_prob = "nan"
                clean_measurements.append(
                    [timestamp, temperature, ph, conductivity, orp, do, label, outlier_prob, event_prob]
                )
            else:
                timestamp = str(unique_timestamp)
                temperature = "nan"
                ph = "nan"
                conductivity = "nan"
                orp = "nan"
                do = "nan"
                label = "-1"
                outlier_prob = "1.0"
                event_prob = "nan"
                clean_measurements.append(
                    [timestamp, temperature, ph, conductivity, orp, do, label, outlier_prob, event_prob]
                )
            unique_timestamp += datetime.timedelta(hours=1)

        fw = open(self.monitor_file, "w")
        header = "MeasurementDateTime,Temperature,pH,Conductivity,ORP,DO,Label,OutlierProb,EventProb\n"
        fw.write(header)
        for cm in clean_measurements:
            line = ",".join(cm) + "\n"
            fw.write(line)
        fw.close()

    def load_monitor_data(self, from_timestamp, to_timestamp, target_label=None):
        timestamps = []
        values = []
        fr = open(self.monitor_file, "r")
        fr.readline()
        for line in fr.readlines():
            m = line.strip().split(",")
            timestamp = datetime.datetime.strptime(m[0], "%Y-%m-%d %H:%M:%S")
            label = m[6]
            if target_label is None:
                if from_timestamp <= timestamp <= to_timestamp:
                    timestamps.append(timestamp)
                    temperature = float(m[1])
                    ph = float(m[2])
                    conductivity = float(m[3])
                    orp = float(m[4])
                    do = float(m[5])
                    label = float(label)
                    outlier_prob = float(m[7])
                    event_prob = float(m[8])
                    values.append([temperature, ph, conductivity, orp, do, label, outlier_prob, event_prob])
            else:
                if from_timestamp <= timestamp <= to_timestamp and label == target_label:
                    timestamps.append(timestamp)
                    temperature = float(m[1])
                    ph = float(m[2])
                    conductivity = float(m[3])
                    orp = float(m[4])
                    do = float(m[5])
                    label = float(label)
                    outlier_prob = float(m[7])
                    event_prob = float(m[8])
                    values.append([temperature, ph, conductivity, orp, do, label, outlier_prob, event_prob])
        return timestamps, values

    def normalize_data(self, values):
        normalized_values = copy.deepcopy(values)
        data = np.array(values, dtype=float)[:, 0:5]
        data_min = np.nanmin(data, 0)
        data_max = np.nanmax(data, 0)
        print data_min
        print data_max
        for i in range(len(values)):
            for j in range(5):
                normalized_values[i][j] = np.abs(values[i][j] - data_min[j]) / np.abs(data_max[j] - data_min[j])
        return normalized_values, data_min, data_max

    def classify_historical_data(self, z_score):
        from_timestamp = self.min_timestamp
        to_timestamp = self.min_timestamp + datetime.timedelta(days=365) - datetime.timedelta(hours=1)
        hist_timestamps, hist_values = self.load_monitor_data(from_timestamp, to_timestamp, "nan")
        hist_data = np.array(hist_values, dtype=float)[:, 0:5]
        mu = np.nanmean(hist_data, axis=0)  # mean of historical data without nan
        sigma = np.nanstd(hist_data, axis=0)  # std of historical data without nan
        for i in range(len(hist_timestamps)):  # label measurement without nan
            flag = False
            for j in range(5):
                z = (np.abs(hist_values[i][j] - mu[j])) / sigma[j]
                if z >= z_score:
                    flag = True
            if flag:
                hist_values[i][5] = -1
                hist_values[i][6] = 1.0
            else:
                hist_values[i][5] = 1
                hist_values[i][6] = 0.0
        fr = open(self.monitor_file, "r")
        header = fr.readline()
        lines = fr.readlines()
        fr.close()
        fw = open(self.monitor_file, "w")  # update monitor file
        fw.write(header)
        for line in lines:
            timestamp = datetime.datetime.strptime(line.strip().split(",")[0], "%Y-%m-%d %H:%M:%S")
            if timestamp in hist_timestamps:
                idx = hist_timestamps.index(timestamp)
                value = hist_values[idx]
                timestamp = str(timestamp)
                temperature = str(value[0])
                ph = str(value[1])
                conductivity = str(value[2])
                orp = str(value[3])
                do = str(value[4])
                label = str(int(value[5]))
                outlier_prob = str(value[6])
                event_prob = str(value[7])
                m = [timestamp, temperature, ph, conductivity, orp, do, label, outlier_prob, event_prob]
                fw.write(",".join(m) + "\n")
            else:
                fw.write(line)
        fw.close()

    def optimize_training_parameters(self, n):
        # data
        from_timestamp = self.min_timestamp
        to_timestamp = self.min_timestamp + datetime.timedelta(days=365) + datetime.timedelta(hours=1)
        train_timestamps, train_values = self.load_monitor_data(from_timestamp, to_timestamp, "1")
        train_data = np.array(train_values)[:, 0:5]

        # parameters
        nu = np.linspace(start=1e-5, stop=1e-2, num=n)
        gamma = np.linspace(start=1e-6, stop=1e-3, num=n)
        opt_diff = 1.0
        opt_nu = None
        opt_gamma = None
        fw = open("training_param.csv", "w")
        fw.write("nu,gamma,diff\n")
        for i in range(len(nu)):
            for j in range(len(gamma)):
                classifier = svm.OneClassSVM(kernel="rbf", nu=nu[i], gamma=gamma[j])
                classifier.fit(train_data)
                label = classifier.predict(train_data)
                p = 1 - float(sum(label == 1.0)) / len(label)
                diff = math.fabs(p-nu[i])
                if diff < opt_diff:
                    opt_diff = diff
                    opt_nu = nu[i]
                    opt_gamma = gamma[j]
                fw.write(",".join([str(nu[i]), str(gamma[j]), str(diff)]) + "\n")
        fw.close()
        return opt_nu, opt_gamma

    def plot_training_parameters(self):
        fr = open("training_param.csv", "r")
        fr.readline()
        lines = fr.readlines()
        fr.close()
        n = 100
        nu = np.empty(n, dtype=np.float64)
        gamma = np.empty(n, dtype=np.float64)
        diff = np.empty([n, n], dtype=np.float64)
        for row in range(len(lines)):
            m = lines[row].strip().split(",")
            i = row / n
            j = row % n
            nu[i] = Decimal(m[0])
            gamma[j] = Decimal(m[1])
            diff[i][j] = Decimal(m[2])
        plt.pcolor(gamma, nu, diff, cmap="coolwarm")
        plt.title("The Difference of Guassian Classifier with Different nu, gamma")
        plt.xlabel("gamma")
        plt.ylabel("nu")
        plt.xscale("log")
        plt.yscale("log")
        plt.colorbar()
        plt.show()

    def predict(self, nu, gamma):
        # classifier
        classifier = svm.OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        # data for test
        from_timestamp = self.min_timestamp + datetime.timedelta(days=365)
        to_timestamp = self.max_timestamp
        test_timestamps, test_values = self.load_monitor_data(from_timestamp, to_timestamp, "nan")
        test_data = np.array(test_values)[:, 0:5]
        # data for train
        to_timestamp = self.min_timestamp + datetime.timedelta(days=365) + datetime.timedelta(hours=1)
        train_timestamps, train_values = self.load_monitor_data(self.min_timestamp, to_timestamp, "1")
        for i in range(len(test_timestamps)):
            # predict
            train_data = np.array(train_values)[:, 0:5]
            classifier.fit(train_data)
            label = classifier.predict(test_data[i])[0]
            test_values[i][5] = int(label)
            if label == 1:
                test_values[i][6] = 0.0
                train_values.append(test_values[i])
            else:
                test_values[i][6] = 1.0
            print test_timestamps[i], label, test_values[i]
        # write result into monitor file
        fr = open(self.monitor_file, "r")
        header = fr.readline()
        lines = fr.readlines()
        fr.close()
        fw = open(self.monitor_file, "w")  # update monitor file
        fw.write(header)
        for line in lines:
            timestamp = datetime.datetime.strptime(line.strip().split(",")[0], "%Y-%m-%d %H:%M:%S")
            if timestamp in test_timestamps:
                idx = test_timestamps.index(timestamp)
                value = test_values[idx]
                timestamp = str(timestamp)
                temperature = str(value[0])
                ph = str(value[1])
                conductivity = str(value[2])
                orp = str(value[3])
                do = str(value[4])
                label = str(int(value[5]))
                outlier_prob = str(value[6])
                event_prob = str(value[7])
                m = [timestamp, temperature, ph, conductivity, orp, do, label, outlier_prob, event_prob]
                fw.write(",".join(m) + "\n")
            else:
                fw.write(line)
        fw.close()

    def estimate_event_probability(self, r, n, p):
        # Binomial Event Discriminator
        from_timestamp = self.min_timestamp + datetime.timedelta(days=365)
        to_timestamp = self.max_timestamp
        timestamps, values = self.load_monitor_data(from_timestamp, to_timestamp, None)
        values = np.array(values, dtype=float)
        prob = math.factorial(n) / (math.factorial(r) * math.factorial(n-r)) * math.pow(p, r) * (math.pow(1-p, n-r))

    def plot_historical_trend(self):
        self.initialize_monitor()
        # plot historical data with outliers
        from_timestamp = self.min_timestamp
        to_timestamp = self.min_timestamp + datetime.timedelta(days=365) - datetime.timedelta(hours=1)
        timestamps, values = self.load_monitor_data(from_timestamp, to_timestamp, None)
        values = np.array(values, dtype=float)
        title = "Historical Data with Outliers"
        #
        temp = values[:, 0]
        ph = values[:, 1]
        cond = values[:, 2]
        orp = values[:, 3]
        do = values[:, 4]
        fig = plt.figure()
        fig.suptitle(self.name + " - " + title)
        #
        ax_temp = fig.add_subplot(511)
        ax_temp.plot(timestamps, temp, "-r")
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_ylim([5, 35])
        #
        ax_ph = fig.add_subplot(512)
        ax_ph.plot(timestamps, ph, "-g")
        ax_ph.set_ylabel("pH")
        ax_ph.set_ylim([0, 14])
        #
        ax_cond = fig.add_subplot(513)
        ax_cond.plot(timestamps, cond, "-b")
        ax_cond.set_ylabel("Conductivity")
        ax_cond.set_ylim([0, 3.5])
        #
        ax_orp = fig.add_subplot(514)
        ax_orp.plot(timestamps, orp, "-m")
        ax_orp.set_ylabel("ORP")
        ax_orp.set_ylim([-1000, 6000])
        #
        ax_do = fig.add_subplot(515)
        ax_do.plot(timestamps, do, "-c")
        ax_do.set_ylabel("DO")
        ax_do.set_xlabel("Date Time")
        ax_do.set_ylim([0, 10])
        #
        plt.show()

        # plot historical data without outliers
        timestamps2 = []
        values2 = []
        timestamps, values = self.load_monitor_data(from_timestamp, to_timestamp, "1")
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        timestamp = min_timestamp
        while timestamp <= max_timestamp:
            if timestamp in timestamps:
                idx = timestamps.index(timestamp)
                values2.append(values[idx])
            else:
                values2.append([float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                               float("nan"), float("nan"), float("nan")])
            timestamp += datetime.timedelta(hours=1)
            timestamps2.append(timestamp)
        values2 = np.array(values2, dtype=float)
        title = "Historical Data without Outliers"
        #
        temp = values2[:, 0]
        ph = values2[:, 1]
        cond = values2[:, 2]
        orp = values2[:, 3]
        do = values2[:, 4]
        fig = plt.figure()
        fig.suptitle(self.name + " - " + title)
        #
        ax_temp = fig.add_subplot(511)
        ax_temp.plot(timestamps2, temp, "-r")
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_ylim([5, 35])
        #
        ax_ph = fig.add_subplot(512)
        ax_ph.plot(timestamps2, ph, "-g")
        ax_ph.set_ylabel("pH")
        ax_ph.set_ylim([0, 14])
        #
        ax_cond = fig.add_subplot(513)
        ax_cond.plot(timestamps2, cond, "-b")
        ax_cond.set_ylabel("Conductivity")
        ax_cond.set_ylim([0, 3.5])
        #
        ax_orp = fig.add_subplot(514)
        ax_orp.plot(timestamps2, orp, "-m")
        ax_orp.set_ylabel("ORP")
        ax_orp.set_ylim([-1000, 6000])
        #
        ax_do = fig.add_subplot(515)
        ax_do.plot(timestamps2, do, "-c")
        ax_do.set_ylabel("DO")
        ax_do.set_ylim([0, 10])
        ax_do.set_xlabel("Date Time")
        #
        plt.show()

    def plot_outlier_trend(self):
        self.initialize_monitor()
        # plot historical data with outliers
        from_timestamp = self.min_timestamp + datetime.timedelta(days=365)
        to_timestamp = self.max_timestamp
        timestamps, values = self.load_monitor_data(from_timestamp, to_timestamp, None)
        values = np.array(values, dtype=float)
        title = "Outlier Detection Results"
        #
        temp = values[:, 0]
        ph = values[:, 1]
        cond = values[:, 2]
        orp = values[:, 3]
        do = values[:, 4]
        labels = values[:, 5]
        fig = plt.figure()
        fig.suptitle(self.name + " - " + title)
        #
        ax_temp = fig.add_subplot(611)
        ax_temp.plot(timestamps, temp, "-r")
        ax_temp.set_ylabel("Temperature")
        #
        ax_ph = fig.add_subplot(612)
        ax_ph.plot(timestamps, ph, "-g")
        ax_ph.set_ylabel("pH")
        #
        ax_cond = fig.add_subplot(613)
        ax_cond.plot(timestamps, cond, "-b")
        ax_cond.set_ylabel("Conductivity")
        #
        ax_orp = fig.add_subplot(614)
        ax_orp.plot(timestamps, orp, "-m")
        ax_orp.set_ylabel("ORP")
        #
        ax_do = fig.add_subplot(615)
        ax_do.plot(timestamps, do, "-c")
        ax_do.set_ylabel("DO")
        #
        ax_lb = fig.add_subplot(616)
        ax_lb.plot(timestamps, labels, "-y")
        ax_lb.set_ylim([-2, 2])
        ax_lb.set_yticks([-1, 1])
        ax_lb.set_ylabel("Label")
        ax_lb.set_xlabel("Date Time")
        #
        plt.show()

    def start(self):
        print ">> time: %s" % time.strftime("%X")
        print ">> initialize monitor %s...\n" % self.name
        self.initialize_monitor()

        print ">> creating the monitor file..."
        self.create_monitor_file()
        print ">> monitor file created!\n"

        print ">> time: %s" % time.strftime("%X")
        print ">> classifying historical data..."
        self.classify_historical_data(z_score=1.98)
        print ">> classification finished!\n"

        print ">> optimize the training parameters..."
        # opt_nu, opt_gamma = self.optimize_training_parameters(n=50)
        print "optimization finished!\n"

        print ">> time: %s" % time.strftime("%X")
        print ">> monitoring new measurements..."
        opt_nu, opt_gamma = 0.001, 0.1
        self.predict(nu=opt_nu, gamma=opt_gamma)
        print ">> monitor stopped!"
        print ">> time: %s" % time.strftime("%X")


if __name__ == "__main__":
    monitor = Monitor(name="CHILL", input_file="chill_realtime.csv")
    # monitor = Monitor(name="ARDEC", input_file="ardec_realtime.csv")
    monitor.start()
    monitor.plot_outlier_trend()
    # monitor.plot_training_parameters()
