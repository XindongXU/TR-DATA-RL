import numpy as np
import csv
import matplotlib.pyplot as plt

CNT = 1000


def read_file(filename):

    rows = np.zeros([CNT, 5])
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            rows[i - 1] = np.array(row)

    rows = np.array(rows).astype(float)
    return rows


def correct_input(rows):

    mask = rows[:, 4] > 0.0
    mask = np.logical_and(mask, rows[:, 3] < 1400.0)

    rows = rows[mask, :]
    return rows


def main():
    rows = read_file("2servos.csv")
    rows = correct_input(rows)

    plt.scatter(rows[:, 3], rows[:, 2])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(rows[:, 3], rows[:, 2], rows[:, 4])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(rows[:, 3], rows[:, 2], rows[:, 4])
    plt.show()

    return 0

if __name__ == "__main__":
    main()
