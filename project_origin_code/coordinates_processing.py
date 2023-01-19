import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from scipy import interpolate
from numpy.polynomial import Polynomial

EPS = 0.005
CNT_FRAMES = 10
VAR = 0.1


def pass_input(filename):

    rows = np.zeros([25000, 5])
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            rows[i] = np.array(row)

    rows = np.array(rows).astype(float)

    theta = rows[:, 4]
    x = rows[:, 0]
    y = rows[:, 1]

    return [x, y, theta]


def correct_input(x, y, theta):
    new_theta = np.zeros(theta.shape)
    for i, t in enumerate(theta):
        dist = min(np.abs(t - math.pi / 2), np.abs(t + math.pi / 2))

        if dist < 0.1:
            new_theta[i] = math.pi / 2 - dist if x[i] > 0.5 else dist - math.pi / 2
        else:
            new_theta[i] = t

    return x, y, new_theta


def processing():
    [y, x, theta] = pass_input("coord.csv")

    x = (x / 1080 - 0.5) * 2
    y = (y / 1920 - 0.5) * 2

    [x, y, theta] = correct_input(x, y, theta)

    num_frames = 15
    theta_avg = np.convolve(theta, np.ones(num_frames), mode='valid') / num_frames

    theta = theta[num_frames // 2: -(num_frames // 2)]
    x = x[num_frames // 2: -(num_frames // 2)]
    y = y[num_frames // 2: -(num_frames // 2)]

    theta_avg_grad = np.gradient(theta_avg)
    mask = np.abs(theta_avg_grad) < EPS

    static_frames_centers = []
    static_frames_length = []
    length = 0
    for i, val in enumerate(mask):
        if val:
            length += 1
            continue

        if length > CNT_FRAMES:
            static_frames_centers.append(i - length // 2)
            static_frames_length.append(length)

        length = 0

    static_frames_length = np.array(static_frames_length)
    static_frames_centers = np.array(static_frames_centers)
    med = np.median(static_frames_length)
    minn = np.min(static_frames_length[1:])

    real_lengths = []
    real_centers = []
    length = 0
    for i, val in enumerate(mask):
        if val:
            length += 1
            continue

        if length > CNT_FRAMES and length < 1.25 * med:
            real_centers.append(i - length // 2)
            real_lengths.append(length)

        elif length > CNT_FRAMES:
            actual_real_frames = np.round(length / med).astype(int)
            actual_length = length // actual_real_frames

            if actual_length < minn:
                actual_real_frames -= 1
                actual_length = length // actual_real_frames

            for j in range(actual_real_frames - 1, -1, -1):
                real_centers.append(i - (actual_length // 2) - j * actual_length)
                real_lengths.append(actual_length)

        length = 0
    return real_centers, real_lengths, x[real_centers], y[real_centers], theta[real_centers]


def poly_fit(x, y, grade=3, hold_on=True):

    coeffs = Polynomial.fit(x, y, grade)
    coeffs = coeffs.convert().coef
    coeffs = coeffs[::-1]

    yfit = np.polyval(coeffs, x)

    plt.plot(x, y, '.', c="g")
    plt.plot(x, yfit, '-', c="r")

    if not hold_on:
        plt.show()

    err = np.sum(np.abs(yfit - y))
    return [coeffs, err]


def spline_fit(x, y, knots=10, hold_on=True):

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

    return theta1_spl, theta2_spl, x1_spl, x2_spl, y1_spl, y2_spl


def func(spline, value):
    def ret(guess):
        return (spline(guess) - value)
    return ret


def inverse_kinematics(spline, value, initial_guess=0.0):

    return fsolve(func(spline, value), x0=initial_guess, xtol=EPS)


def main():

    servo_angles = [-75, 75, 57, -35, 22, -71, -14, 44, -49, 8, 66,
                    -27, 30, -62, -5, 52, -41, 17, 74, -19, 39, -54,
                    3, 60, -32, 25, -68, -10, 47, -46, 12, 69, 44, 3,
                    25, 22, -32, -75, -35, -27, 66, -71, 69, 57, 12,
                    -41, 30, 39, 52, -19, -62, -10, 75, -5, 17, -68,
                    74, -54, 60, 47, -46, 8, -49, -14, 74, -75, -49,
                    -62, -14, 44, -71, 17, -19, -19, 22, 60, 25, -71,
                    66, 3, 60, -41, -41, 12, 66, 52, 74, 30, 8, 17, 75,
                    -46, -75, -14, 47, -68, 39, 52, -5, -54, -27, -32,
                    -49, -27, -5, -62, 57, -10, 3, -35, -32, 75, -35, 47,
                    -54, 30, 69, -46, -10, -68, 39, 8, 44, 25, 69, 22, 12,
                    57, -41, 60, 22, 12, 74, -46, 57, 75, -54, 39, -32,
                    -10, -35, 47, -19, 74, 17, 17, 8, -19, -41, -5, 60,
                    69, 39, -14, -41, -54, -75, 52, -19, -68, 75, 44,
                    -32, -5, 25, -27, -35, 3, 3, 75, 8, 22, -46, 74,
                    -68, 60, 30, -54, 75, -41, 22, 25, 30, -27, 69, -32,
                    -71, 39, -10, -14, -62, 66, -71, 66, -10, 17, 8, 25,
                    39, 17, -62, 12, -75, 60, 47, 3, 44, -14, -49, 57, 3,
                    -62, 22, -49, -27, -68, 52, 44, -71, 74, 25, 47, -5, -68,
                    -14, 44, -46, -49, 8, 30, -27, 52, -46, -75, 57, -5, 12,
                    -49, 52, 69, 47, -62, 66, 30, 66, -75, 12, -19, -35, -10,
                    69, 57, -32, -35, -54, -71]

    servo_angles = np.array(servo_angles)

    stable_frames, stable_lengths, x, y, theta = processing()

    stable_frames = stable_frames[1:len(servo_angles) + 1]
    stable_lengths = stable_lengths[1:len(servo_angles) + 1]
    x, y = x[1:len(servo_angles) + 1], y[1:len(servo_angles) + 1]
    theta = theta[1:len(servo_angles) + 1]

    aux = forward_kinematics(servo_angles, x, y, theta, show_images=True)
    theta1_spl, theta2_spl, x1_spl, x2_spl, y1_spl, y2_spl = aux

    a1 = []
    a2 = []
    max_err = 0.0
    for i, val in enumerate(x):
        val1 = inverse_kinematics(x1_spl, val)
        a1.append(val1)
        val2 = inverse_kinematics(x2_spl, val)
        a2.append(val2)
        err = min(np.abs(val1 - servo_angles[i]), np.abs(val2 - servo_angles[i]))
        max_err = max(max_err, err)
        print(str(servo_angles[i]) + ": " + str(val1) + str(val2) + str(err))

    print(max_err)

    plt.scatter(theta, servo_angles, label="original")
    plt.scatter(theta, a1, label="dir1")
    plt.scatter(theta, a2, label="dir2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
