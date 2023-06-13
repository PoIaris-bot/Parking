import cv2
import time
import json
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def constraint(value, lb, ub):
    if value > ub:
        return ub
    if value < lb:
        return lb
    return value


def map_value(value, lb1, ub1, lb2, ub2):
    value = constraint(value, lb1, ub1)
    value = (value - lb1) / (ub1 - lb1) * (ub2 - lb2) + lb2
    return constraint(value, lb2, ub2)


def remap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


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


def calc_lane_path(lane_contour, img_sz):
    lane_area = np.zeros((img_sz, img_sz), dtype=np.uint8)
    cv2.drawContours(lane_area, [lane_contour], -1, 255, cv2.FILLED)
    _, lane_area = cv2.threshold(lane_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lane_area = cv2.morphologyEx(lane_area, cv2.MORPH_OPEN, np.ones((10, 10), dtype=np.uint8), iterations=5)
    skeleton = skeletonize(lane_area, method='lee')
    skeleton = np.where(skeleton > 0)
    skeleton = np.column_stack((skeleton[1], skeleton[0]))
    distance_matrix = np.linalg.norm(skeleton[:, None] - skeleton, axis=2).astype(np.int32)
    end_idx = np.where((distance_matrix <= 1).sum(axis=0) == 2)[0].tolist()

    last_idx = None
    cur_idx = end_idx[0] if skeleton[end_idx[0]][0] > skeleton[end_idx[1]][0] else end_idx[1]
    lane_path = [skeleton[cur_idx, :]]
    for i in range(skeleton.shape[0] - 1):
        dist = distance_matrix[cur_idx, :]
        nearest = np.where(dist <= 1)[0].tolist()
        if last_idx is not None:
            nearest.remove(last_idx)
        nearest.remove(cur_idx)
        last_idx = cur_idx
        cur_idx = nearest[0]
        lane_path.append(skeleton[cur_idx, :])
    lane_path = np.array(lane_path)
    sample_idx = np.linspace(0, lane_path.shape[0] - 1, int(50 / img_sz * 800), endpoint=True).astype(np.int32)
    return lane_path[sample_idx, :]


def calc_lot_center(spot_contour):
    x, y, w, h = cv2.boundingRect(spot_contour)
    return x + w // 2, y + h // 2


def calc_map(area_contours, img_sz):
    scale = img_sz / 800
    lot_contour, lane_contour, spot1_contour, spot2_contour = area_contours
    lane_path = calc_lane_path(lane_contour, img_sz)
    spot1_point = calc_lot_center(spot1_contour)
    spot2_point = calc_lot_center(spot2_contour)

    entrance_point = lane_path[-1, :].tolist()
    exit_point = lane_path[0, :].tolist()
    switch1_point = [spot1_point[0], exit_point[1]]
    switch2_point = [spot2_point[0], exit_point[1]]

    delta_x1, delta_y1 = abs(entrance_point[0] - switch1_point[0]), abs(entrance_point[1] - switch1_point[1]) * 0.8
    delta_x2, delta_y2 = abs(entrance_point[0] - switch2_point[0]), abs(entrance_point[1] - switch2_point[1]) * 0.8
    y1 = np.linspace(0, delta_y1, int(20 * scale))
    y2 = np.linspace(0, delta_y2, int(25 * scale))
    x1 = delta_x1 / 2 * (np.cos(y1 / delta_y1 * np.pi) - 1)
    x2 = delta_x2 / 2 * (np.cos(y2 / delta_y2 * np.pi) - 1)
    spot1_in_fw_path1 = np.array([entrance_point[0] - x1, entrance_point[1] - y1]).T
    spot2_in_fw_path1 = np.array([entrance_point[0] - x2, entrance_point[1] - y2]).T
    spot1_in_fw_path2 = np.linspace(
        switch1_point, [switch1_point[0], entrance_point[1] - delta_y1], int(4 * scale), endpoint=False
    )
    spot2_in_fw_path2 = np.linspace(
        switch2_point, [switch2_point[0], entrance_point[1] - delta_y2], int(4 * scale), endpoint=False
    )
    spot1_in_fw_path = np.vstack([spot1_in_fw_path1, np.flipud(spot1_in_fw_path2)])
    spot2_in_fw_path = np.vstack([spot2_in_fw_path1, np.flipud(spot2_in_fw_path2)])

    spot1_in_bw_path = np.linspace(switch1_point, spot1_point, int(15 * scale))
    spot2_in_bw_path = np.linspace(switch2_point, spot2_point, int(15 * scale))

    r1 = abs(exit_point[0] - spot1_point[0]) * 0.7
    angle1 = np.linspace(np.pi, np.pi / 2, int(15 * scale))
    spot1_out_path1 = np.linspace(
        spot1_point, [spot1_point[0], exit_point[1] + r1], int(7 * scale), endpoint=False
    )
    spot1_out_path2 = np.array([spot1_point[0] + r1 + r1 * np.cos(angle1), exit_point[1] + r1 - r1 * np.sin(angle1)])
    spot1_out_path3 = np.linspace(
        exit_point, [spot1_point[0] + r1, exit_point[1]], int(5 * scale), endpoint=False
    )
    spot1_out_path = np.vstack([spot1_out_path1, spot1_out_path2.T, np.flipud(spot1_out_path3)])

    r21 = abs(spot1_point[0] - spot2_point[0])
    angle21 = np.linspace(0, np.pi / 2, int(10 * scale), endpoint=False)
    spot2_out_path1 = np.array([spot1_point[0] + r21 * np.cos(angle21), spot1_point[1] - r21 * np.sin(angle21)])
    r22 = abs((spot1_point[1] - r21 - exit_point[1]) / 2)
    angle22 = np.linspace(3 * np.pi / 2, np.pi / 2, int(20 * scale), endpoint=False)
    spot2_out_path2 = np.array([spot1_point[0] + r22 * np.cos(angle22), exit_point[1] + r22 - r22 * np.sin(angle22)])
    spot2_out_path3 = np.linspace([spot1_point[0], exit_point[1]], exit_point, int(10 * scale))
    spot2_out_path = np.vstack([spot2_out_path1.T, spot2_out_path2.T, spot2_out_path3])

    return {
        'contour': {
            'lot': lot_contour,
            'lane': lane_contour,
            'spot1': spot1_contour,
            'spot2': spot2_contour
        },
        'path': {
            'lane': lane_path,
            'spot1_in_fw': spot1_in_fw_path,
            'spot1_in_bw': spot1_in_bw_path,
            'spot1_out': spot1_out_path,
            'spot2_in_fw': spot2_in_fw_path,
            'spot2_in_bw': spot2_in_bw_path,
            'spot2_out': spot2_out_path
        }
    }


def save_map_to_json(parking_map):
    parking_map_copy = {}
    for key1 in parking_map.keys():
        parking_map_copy[key1] = {}
        for key2 in parking_map[key1].keys():
            parking_map_copy[key1][key2] = parking_map[key1][key2].tolist()
    with open(ROOT / 'map.json', 'w') as file:
        json.dump(parking_map_copy, file, indent=4)


def preprocess(frame, img_sz):
    raw_sz = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)

    i, count = 0, 0
    area_centers = []
    area_contours = []
    for i, contour in enumerate(contours):  # find parking lot and lane areas
        x, y, w, h = cv2.boundingRect(contour)
        if x not in [0, raw_sz[1] - w] and y not in [0, raw_sz[0] - h]:
            area_contours.append(contour)
            m = cv2.moments(contour)
            xc = int(m['m10'] / m['m00'])
            yc = int(m['m01'] / m['m00'])
            area_centers.append([xc, yc])
            count += 1
        if count == 2:
            break
    convex_hull = np.zeros_like(gray, dtype=np.uint8)
    convex_hull_contour = cv2.convexHull(np.vstack(area_contours))
    cv2.drawContours(convex_hull, [convex_hull_contour], -1, 255, cv2.FILLED)

    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(convex_hull).squeeze().tolist()
    lines.sort(key=lambda line: np.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2), reverse=True)
    lines = sorted(lines[:4], key=lambda line: np.arctan2(abs(line[1] - line[3]), abs(line[0] - line[2])))

    pt1 = calc_line_cross_point(lines[0], lines[2])
    pt2 = calc_line_cross_point(lines[0], lines[3])
    pt3 = calc_line_cross_point(lines[1], lines[2])
    pt4 = calc_line_cross_point(lines[1], lines[3])

    pts = np.array([pt1, pt2, pt3, pt4])
    pts = pts[pts[:, 0].argsort()]
    top_left, bottom_left = (pts[0], pts[1]) if pts[0, 1] < pts[1, 1] else (pts[1], pts[0])
    top_right, bottom_right = (pts[2], pts[3]) if pts[2, 1] < pts[3, 1] else (pts[3], pts[2])

    matrix_p = cv2.getPerspectiveTransform(
        np.float32([top_left, top_right, bottom_left, bottom_right]),
        np.float32([[0, 0], [img_sz, 0], [0, img_sz], [img_sz, img_sz]])
    )

    angle_dict = {1: 270, 2: 180, -1: 0, -2: 90}
    angle = np.arctan2(area_centers[0][1] - area_centers[1][1], area_centers[1][0] - area_centers[0][0])
    quadrant = int(angle / np.pi * 2) + (1 if angle > 0 else -1)

    matrix_r = cv2.getRotationMatrix2D(
        (img_sz // 2, img_sz // 2), angle_dict[quadrant], 1
    )
    for contour in contours[i + 1:]:  # find two parking spot areas
        x, y, w, h = cv2.boundingRect(contour)
        if x not in [0, raw_sz[0] - w] and y not in [0, raw_sz[1] - h]:
            if cv2.pointPolygonTest(convex_hull_contour, (x + w // 2, y + h // 2), False) > 0:
                if cv2.contourArea(contour) > w * h * 0.5:
                    area_contours.append(contour)
                    count += 1
            if count == 4:
                break

    areas = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(areas, area_contours, -1, 255, cv2.FILLED)
    areas = cv2.warpPerspective(areas, matrix_p, (img_sz, img_sz))
    areas = cv2.warpAffine(areas, matrix_r, (img_sz, img_sz))

    _, areas = cv2.threshold(areas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area_contours, _ = cv2.findContours(areas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)

    area_contours = [contour.squeeze() for contour in area_contours]
    if area_contours[2][0, 0] > area_contours[3][0, 0]:
        area_contours[2], area_contours[3] = area_contours[3], area_contours[2]
    parking_map = calc_map(area_contours[:4], img_sz)

    return matrix_p, matrix_r, parking_map


def load_map_from_json():
    with open(ROOT / 'map.json', 'r') as file:
        parking_map = json.load(file)
    for key1 in parking_map.keys():
        for key2 in parking_map[key1].keys():
            parking_map[key1][key2] = np.array(parking_map[key1][key2])

    return parking_map


def init_pos_check(parking_map, cars, scale):
    for key in cars.keys():
        x, y, theta = cars[key]
        if cv2.pointPolygonTest(parking_map['contour']['lot'], (x, y), False) > 0:
            return False
        if cv2.pointPolygonTest(parking_map['contour']['spot1'], (x, y), False) > 0:
            if abs(theta + np.pi / 2) > np.pi / 12:
                return False
        if cv2.pointPolygonTest(parking_map['contour']['spot2'], (x, y), False) > 0:
            if abs(theta + np.pi / 2) > np.pi / 12:
                return False
        if cv2.pointPolygonTest(parking_map['contour']['lane'], (x, y), False) > 0:
            entrance_point, exit_point = parking_map['path']['lane'][[-1, 0], :]
            if np.linalg.norm(entrance_point - np.array([x, y]).T) < 50 * scale:
                if abs(theta + np.pi / 2) > np.pi / 12:
                    return False
            elif np.linalg.norm(exit_point - np.array([x, y]).T) < 50 * scale:
                if abs(theta) > np.pi / 12:
                    return False
            else:
                return False
    return True


class StateMachine:
    def __init__(self, init_pose, parking_map, img_sz):
        self.paths = {
            '1': parking_map['path']['lane'],
            '2': None,
            '3-1': parking_map['path']['spot1_in_fw'],
            '3-2': parking_map['path']['spot2_in_fw'],
            '4-1': parking_map['path']['spot1_in_bw'],
            '4-2': parking_map['path']['spot2_in_bw'],
            '5-1': None,
            '5-2': None,
            '6-1': parking_map['path']['spot1_out'],
            '6-2': parking_map['path']['spot2_out'],
            '7': None
        }
        self.brake_flag = False
        self.brake_count = 0

        self.img_sz = img_sz
        self.scale = img_sz / 800
        self.fw_controller = PIDController(2, 0, 4)
        self.bw_controller = PIDController(3, 0, 5)

        if cv2.pointPolygonTest(parking_map['contour']['spot1'], init_pose[:2], False) > 0:
            self.state = '5-1'
        if cv2.pointPolygonTest(parking_map['contour']['spot2'], init_pose[:2], False) > 0:
            self.state = '5-2'
        if cv2.pointPolygonTest(parking_map['contour']['lane'], init_pose[:2], False) > 0:
            entrance_point, exit_point = parking_map['path']['lane'][[-1, 0], :]
            if np.linalg.norm(entrance_point - np.array(init_pose[:2])) < 50 * self.scale:
                self.state = '2'
            elif np.linalg.norm(exit_point - np.array(init_pose[:2])) < 50 * self.scale:
                self.state = '7'

    def update(self, pose, states):
        x, y, theta = pose
        if self.state == '1':
            if np.linalg.norm(self.paths[self.state][-1, :] - np.array([x, y])) < 50 * self.scale:
                self.state = '2'
        elif self.state == '2':
            if not {'3-1', '3-2', '4-1', '4-2', '6-1', '6-2'} & set(states):
                if '5-1' not in states:
                    self.state = '3-1'
                elif '5-2' not in states:
                    self.state = '3-2'
        elif self.state[0] == '3':
            if np.linalg.norm(self.paths[self.state][-1, :] - np.array([x, y])) < 50 * self.scale:
                self.state = '4' + self.state[1:]
        elif self.state[0] == '4':
            if abs(self.paths[self.state][-1, 1] - y) < 50 * self.scale:
                self.state = '5' + self.state[1:]
                self.brake_flag, self.brake_count = True, 0
        elif self.state[0] == '5':
            if not {'3-1', '3-2', '4-1', '4-2', '6-1', '6-2'} & set(states):
                if '7' not in states:
                    if self.state[-1] == '1':
                        self.state = '6-1'
                    else:  # 2
                        self.state = '6-2'
                    self.brake_flag, self.brake_count = False, 0
        elif self.state[0] == '6':
            if np.linalg.norm(self.paths[self.state][-1, :] - np.array([x, y])) < 50 * self.scale:
                self.state = '7'
        else:  # 7
            if '1' not in states:
                self.state = '1'

        path = self.paths[self.state]
        if path is None:
            if self.brake_flag:
                self.brake_count += 1
                if self.brake_count >= 10:
                    self.brake_flag = False
                return [90, 1, 20]
            else:
                return [90, 0, 0]
        else:
            direction = 0 if self.state[0] == '4' else 1
            theta = remap_angle(-theta + np.pi) if direction == 0 else -theta

            distance = np.linalg.norm(np.array([[x, y]]) - path, axis=1)
            closest_idx = np.argmin(distance)
            try:
                xd, yd = path[closest_idx + (3 if direction == 1 else 4), :]
                d = distance[closest_idx + (3 if direction == 1 else 4)]
            except IndexError:
                xd, yd = path[-1, :]
                d = distance[-1]
            y = self.img_sz - y
            yd = self.img_sz - yd

            theta_d = np.arctan2(yd - y, xd - x)
            theta_e = remap_angle(theta_d - theta)
            u = self.fw_controller.output(theta_e) if direction == 1 else self.bw_controller.output(-theta_e)
            angle = 90 + 15 * np.tanh(u)

            speed = 13 * np.tanh(0.05 * d)
            return [angle, direction, speed]


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.e = 0
        self.e_sum = 0

    def output(self, e):
        de = e - self.e
        self.e_sum = self.e_sum + e
        self.e = e
        return self.kp * e + self.ki * self.e_sum + self.kd * de
