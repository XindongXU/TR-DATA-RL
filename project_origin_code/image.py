import cv2
import numpy as np
import find_stick
import matplotlib.pyplot as plt
from sklearn import linear_model
from tqdm import tqdm

STICK_COLOR = [255, 0, 255]
HEIGHT = 1920
WIDTH = 1080


def sort_cmp(l1):
    return -(l1[0] - WIDTH / 2) ** 2 - (l1[1] - HEIGHT / 2) ** 2


def find_base_old(x_predict, y_predict, k):

    ll = list(zip(x_predict, y_predict))
    ll.sort(key=sort_cmp)

    return np.array(ll[:k]).mean(axis=0).squeeze()


def find_base(x_predict, y_predict, k):
    ll = np.array(list(zip(x_predict, y_predict)))

    sigma = np.std(ll, axis=0)
    center = np.mean(ll, axis=0)

    maskx = np.abs(ll[:, 0] - center[0]) < sigma[0]
    masky = np.abs(ll[:, 1] - center[1]) < sigma[1]

    mask = np.logical_and(maskx, masky)

    x_predict = x_predict[mask]
    y_predict = y_predict[mask]

    # print(x_predict.shape)
    return x_predict.mean(), y_predict.mean()


def liniar_regression_ransac(x, y):

    ransac = linear_model.RANSACRegressor(min_samples = 2.0, stop_n_inliers = 1000, max_trials = 2000, residual_threshold = 4.0)
    ransac.fit(x, y)

    s = ransac.score(x, y)
    m = float(ransac.estimator_.coef_)
    c = float(ransac.estimator_.intercept_)

    inlier_mask = ransac.inlier_mask_

    return [s, m, c, inlier_mask]


def best_liniar_regression(x, y):

    [s1, m1, c1, mask1] = liniar_regression_ransac(x, y)
    [s2, m2, c2, mask2] = liniar_regression_ransac(y, x)

    if np.count_nonzero(mask1) > np.count_nonzero(mask2):
        return [s1, m1, c1, mask1]

    return [s2, 1 / m2, -c2 / m2, mask2]


def apply_stick_pixels(x, y, mask, image):

    x_predict = x[mask]
    y_predict = y[mask]

    image[x_predict, y_predict, :] = STICK_COLOR

    return image


def get_points(m, c):

    xlin = np.linspace(0, HEIGHT - 1, 300)
    ylin = m * xlin + c

    in_image = np.logical_and(ylin < WIDTH - 1, ylin >= 0)
    xlin = xlin[in_image]
    ylin = ylin[in_image]

    return [xlin, ylin]


def plot_liniar_regression(x, y, mask, image, m, c):

    x_predict = x[mask]
    y_predict = y[mask]

    image[x_predict, y_predict, :] = STICK_COLOR

    plt.imshow(image)

    [xlin, ylin] = get_points(m, c)

    base_x, base_y = find_base(x_predict, y_predict, 5)

    plt.plot(base_y, base_x, color='b', marker='o')
    plt.plot(ylin, xlin, color='red', linewidth=3)
    plt.show()


def identify_stick_video(name, startFrameNumber, finishFrameNumber):

    video = cv2.VideoCapture(name)

    rows = np.array([0] * 5)
    for i in tqdm(range(startFrameNumber, finishFrameNumber)):

        frame_nr = i
        video.set(1, frame_nr)

        succes, frame = video.read()
        frame = cv2.flip(frame, 0)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = find_stick.find_stick_HSV(frame)
        coordinates = np.argwhere(mask)

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        [s, m, c, inlier_mask] = best_liniar_regression(x.reshape(-1, 1), y.reshape(-1, 1))

        base_x, base_y = find_base(x[inlier_mask], y[inlier_mask], 5)

        # frame = apply_stick_pixels(x, y, inlier_mask, frame)
        # plot_liniar_regression(x, y, inlier_mask, frame, m, c)

        row = np.array([base_x, base_y, m, c, np.arctan(m)])
        rows = np.vstack((rows, row.T))

    rows = rows[1:, :]
    # fields = "base_x, base_y. slope, c coeff, theta"
    # np.savetxt("coord.csv", rows, delimiter=",", header=fields, fmt='%.7f', comments='')


if __name__ == "__main__":

    identify_stick_video("tentacle.mp4", 5000, 6026)
