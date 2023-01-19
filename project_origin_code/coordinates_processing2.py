import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate
from numpy.polynomial import Polynomial


CNT = 5000
EPS = 0.05


def parse_input(filename):

    rows = np.zeros([CNT, 4])
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for i, row in enumerate(csvreader):
            if i == 0:
                continue

            if i >= CNT:
                break

            rows[i] = np.array(row[0:4])

    rows = np.array(rows).astype(float)

    servo_angles = rows[1:i, 0]
    theta = rows[1:i, 3]
    x = rows[1:i, 1]
    y = rows[1:i, 2]

    print(y)

    return [servo_angles, y, x, theta]


def poly_fit(x, y, grade=3, hold_on=True):

    coeffs = Polynomial.fit(x, y, grade)
    coeffs = coeffs.convert().coef
    coeffs = coeffs[::-1]

    yfit = np.polyval(coeffs, x)

    plt.plot(x, y, '.', c="g", label="original")
    plt.plot(x, yfit, '-', c="r", label="polyfit")

    if not hold_on:
        plt.show()

    err = np.sum(np.abs(yfit - y))
    return [coeffs, err]


def spline_fit(x, y, knots=30, hold_on=True):

    x_new = np.linspace(0, 1, knots + 2)[1:-1]
    q_knots = np.quantile(x, x_new)

    t, c, k = interpolate.splrep(x, y, t=q_knots, s=1)

    # yfit = interpolate.BSpline(t,c, k)(x)
    b = interpolate.BSpline(t, c, k)
    yfit = b(x)

    plt.plot(x, y, '.', c="g", label="original")
    plt.plot(x, yfit, '-', c="r", label="spline fit")

    if not hold_on:
        plt.show()

    err = np.sum(np.abs(yfit - y))
    return [b, err]


# theta correction
def correct_input(x, y, theta):
    new_theta = np.zeros(theta.shape)
    for i, t in enumerate(theta):
        dist = min(np.abs(t - np.pi / 2), np.abs(t + np.pi / 2))

        if dist < 0.2:
            new_theta[i] = np.pi / 2 - dist if x[i] > 0.9 else dist - np.pi / 2
        else:
            new_theta[i] = t

    return [x, y, new_theta]


def process_input(servo_angles, x, y, theta):

    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    [x, y, theta] = correct_input(x, y, theta)
    # theta = (theta - np.min(theta)) / (np.max(theta) - np.min(theta))

    mask = np.logical_and(x < EPS, y < EPS)
    x = x[~mask]
    y = y[~mask]
    theta = theta[~mask]
    servo_angles = servo_angles[~mask]

    return [servo_angles, x, y, theta]


def forward_kinematics(servo_angles, x, y, theta, show_images=False):

    # Position and orientation distribution

    unique_angles = np.array(list(set(servo_angles)))
    for angle in unique_angles:
        mask = servo_angles == angle
        plt.scatter(x[mask], y[mask], label=f"angle={angle}")
    plt.legend()
    if show_images:
        plt.show()
    else:
        plt.clf()

    diff_servo = np.diff(servo_angles, prepend=0)
    dir1 = diff_servo < 0
    dir2 = diff_servo > 0

    # Sorting servo angles

    sort_mask = np.argsort(servo_angles)

    servo_angles = servo_angles[sort_mask]
    dir1 = dir1[sort_mask]
    dir2 = dir2[sort_mask]
    x = x[sort_mask]
    y = y[sort_mask]
    theta = theta[sort_mask]

    # Cubic polynomial fitting

    coeffs_theta1, err = poly_fit(servo_angles[dir1], theta[dir1])
    print("theta1_pol: " + str(err))
    coeffs_theta2, err = poly_fit(servo_angles[dir2], theta[dir2], hold_on=(not show_images))
    print("theta2_pol: " + str(err))

    coeffs_x1, err = poly_fit(servo_angles[dir1], x[dir1])
    print("x1_pol: " + str(err))
    coeffs_x2, err = poly_fit(servo_angles[dir2], x[dir2], hold_on=(not show_images))
    print("x2_pol: " + str(err))

    coeffs_y1, err = poly_fit(servo_angles[dir1], y[dir1])
    print("y1_pol: " + str(err))
    coeffs_y2, err = poly_fit(servo_angles[dir2], y[dir2], hold_on=(not show_images))
    print("y2_pol: " + str(err))

    # Spline fitting

    theta1_spl, err = spline_fit(servo_angles[dir1], theta[dir1])
    print("theta1_spl_err: " + str(err))
    theta2_spl, err = spline_fit(servo_angles[dir2], theta[dir2], hold_on=(not show_images))
    print("theta2_spl_err: " + str(err))

    x1_spl, err = spline_fit(servo_angles[dir1], x[dir1])
    print("x1_spl_err: " + str(err))
    x2_spl, err = spline_fit(servo_angles[dir2], x[dir2], hold_on=(not show_images))
    print("x2_spl_err: " + str(err))

    y1_spl, err = spline_fit(servo_angles[dir1], y[dir1])
    print("y1_spl_err: " + str(err))
    y2_spl, err = spline_fit(servo_angles[dir2], y[dir2], hold_on=(not show_images))
    print("y2_spl_err: " + str(err))

    print("\n")

    if not show_images:
        plt.clf()

    return [theta1_spl, theta2_spl, x1_spl, x2_spl, y1_spl, y2_spl]


def func(spline, value):
    def ret(guess):
        return (spline(guess) - value)
    return ret


def inverse_kinematics(spline, value, initial_guess=0.5):
    return fsolve(func(spline, value), x0=initial_guess, xtol=EPS)


def get_splines():
    [servo_angles, x, y, theta] = parse_input("test.csv")
    [servo_angles, x, y, theta] = process_input(servo_angles, x, y, theta)

    aux = forward_kinematics(servo_angles, x, y, theta)
    [theta1_spl, theta2_spl, x1_spl, x2_spl, y1_spl, y2_spl] = aux

    return [theta1_spl, theta2_spl, x1_spl, x2_spl, y1_spl, y2_spl]


def main():
    [servo_angles, x, y, theta] = parse_input("test.csv")
    [servo_angles, x, y, theta] = process_input(servo_angles, x, y, theta)

    aux = forward_kinematics(servo_angles, x, y, theta, True)
    [theta1_spl, theta2_spl, x1_spl, x2_spl, y1_spl, y2_spl] = aux

    x1_inv = np.zeros(len(x))
    x2_inv = np.zeros(len(x))
    for i, xp in enumerate(x):
        x1_inv[i] = inverse_kinematics(x1_spl, xp)
        x2_inv[i] = inverse_kinematics(x2_spl, xp)

    plt.scatter(x, servo_angles, c='r', label='original')
    plt.scatter(x, x1_inv, c='g', label='dir1_inv')
    plt.scatter(x, x2_inv, c='b', label='dir2_inv')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
