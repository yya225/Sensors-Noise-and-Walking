import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal

# investigate and plot individual dataset
def main():
    # read dataset from input directory
    data = pd.read_csv(sys.argv[1])

    # plot x data from 10s to 16s
    data.iloc[1000:1600].plot(kind = 'line',
            x = 'time',
            y = 'ax',
            color = 'blue')

    # filter the x data from 10 to 16s
    bx, ax = signal.butter(3, 0.12, btype='lowpass', analog=False)
    xlow = signal.filtfilt(bx, ax, data["ax"].iloc[1000:1600])

    # plot filter data
    plt.plot(data["time"].iloc[1000:1600],xlow,color="green")

    plt.title('data-compare')
    plt.show()

    # # uncomment if want to see y data
    # # plot y data
    # data.iloc[1000:1600].plot(kind = 'line',
    #           x = 'time',
    #           y = 'ay',
    #           color = 'blue')
    #
    #
    # by, ay = signal.butter(3, 0.07, btype='lowpass', analog=False)
    # ylow = signal.filtfilt(by, ay, data["ay"].iloc[1000:1600])
    # plt.plot(data["time"].iloc[1000:1600],ylow,color = "green")
    #
    # plt.title('data-compare')
    # plt.show()


    # # plot z data
    # data.iloc[1000:1600].plot(kind = 'line',
    #           x = 'time',
    #           y = 'az',
    #           color = 'blue')
    #
    # bz, az = signal.butter(3, 0.17, btype='lowpass', analog=False)
    # zlow = signal.filtfilt(bz, az, data["az"].iloc[1000:1600])
    # plt.plot(data["time"].iloc[1000:1600],zlow,color = "green")
    #
    # plt.title('data-compare')
    # plt.show()


if __name__ == '__main__':
    main()
