import numpy as np


def calc_line_param(x1, y1, x2, y2):
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def calc_line_cross_point(line1, line2):
    a1, b1, c1 = calc_line_param(*line1)
    a2, b2, c2 = calc_line_param(*line2)
    d = a1 * b2 - a2 * b1

    x = round((b1 * c2 - b2 * c1) / d)
    y = round((a2 * c1 - a1 * c2) / d)
    return x, y


def calc_pose(corners):
    x, y = corners.mean(axis=0)
    head = (corners[0] + corners[1]) / 2
    rear = (corners[2] + corners[3]) / 2
    theta = np.arctan2(head[1] - rear[1], head[0] - rear[0])
    sz = np.linalg.norm(head - rear)
    return x, y, theta, sz
