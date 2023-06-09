#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import json
import rospy
import numpy as np
import cv2.aruco as aruco
from pathlib import Path
from std_srvs.srv import SetBool
from std_msgs.msg import Float32MultiArray
from skimage.morphology import skeletonize

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


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
    lot_contour, lane_contour, spot1_contour, spot2_contour = area_contours
    lane_path = calc_lane_path(lane_contour, img_sz)
    spot1_point = calc_lot_center(spot1_contour)
    spot2_point = calc_lot_center(spot2_contour)

    entrance_point = lane_path[-1, :].tolist()
    exit_point = lane_path[0, :].tolist()
    switch1_point = [spot1_point[0], exit_point[1]]
    switch2_point = [spot2_point[0], exit_point[1]]

    delta_x1, delta_y1 = abs(entrance_point[0] - switch1_point[0]), abs(entrance_point[1] - switch1_point[1])
    delta_x2, delta_y2 = abs(entrance_point[0] - switch2_point[0]), abs(entrance_point[1] - switch2_point[1])
    y1 = np.linspace(0, delta_y1, int(20 / 800 * img_sz))
    y2 = np.linspace(0, delta_y2, int(25 / 800 * img_sz))
    x1 = delta_x1 / 2 * (np.cos(y1 / delta_y1 * np.pi) - 1)
    x2 = delta_x2 / 2 * (np.cos(y2 / delta_y2 * np.pi) - 1)
    spot1_in_fw_path = np.array([entrance_point[0] - x1, entrance_point[1] - y1]).T
    spot2_in_fw_path = np.array([entrance_point[0] - x2, entrance_point[1] - y2]).T
    spot1_in_bw_path = np.linspace(switch1_point, spot1_point, int(15 / 800 * img_sz))
    spot2_in_bw_path = np.linspace(switch2_point, spot2_point, int(15 / 800 * img_sz))

    r1 = abs(exit_point[0] - spot1_point[0])
    angle1 = np.linspace(np.pi, np.pi / 2, int(15 / 800 * img_sz))
    spot1_out_path1 = np.linspace(
        spot1_point, [spot1_point[0], exit_point[1] + r1], int(5 / 800 * img_sz), endpoint=False
    )
    spot1_out_path2 = np.array([exit_point[0] + r1 * np.cos(angle1), exit_point[1] + r1 - r1 * np.sin(angle1)])
    spot1_out_path = np.vstack([spot1_out_path1, spot1_out_path2.T])

    r21 = abs(spot1_point[0] - spot2_point[0])
    angle21 = np.linspace(0, np.pi / 2, int(10 / 800 * img_sz), endpoint=False)
    spot2_out_path1 = np.array([spot1_point[0] + r21 * np.cos(angle21), spot1_point[1] - r21 * np.sin(angle21)])
    r22 = abs((spot1_point[1] - r21 - exit_point[1]) / 2)
    angle22 = np.linspace(3 * np.pi / 2, np.pi / 2, int(20 / 800 * img_sz), endpoint=False)
    spot2_out_path2 = np.array([spot1_point[0] + r22 * np.cos(angle22), exit_point[1] + r22 - r22 * np.sin(angle22)])
    spot2_out_path3 = np.linspace([spot1_point[0], exit_point[1]], exit_point, int(10 / 800 * img_sz))
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


class Camera:
    def __init__(self):
        rospy.init_node('camera', anonymous=True)

        node_name = rospy.get_name()
        cam = rospy.get_param(node_name + '/cam')
        self.img_sz = rospy.get_param(node_name + '/img_sz')
        self.view_path = rospy.get_param(node_name + '/view_path')
        self.view_contour = rospy.get_param(node_name + '/view_contour')

        self.cap = cv2.VideoCapture(cam, cv2.WINDOW_NORMAL)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        self.cars = {}
        self.pose_pub = rospy.Publisher('/pose', Float32MultiArray, queue_size=1)

        ret = False
        while not ret:
            ret, frame = self.cap.read()
            if ret:
                self.matrix_p, self.matrix_r, self.parking_map = preprocess(frame, self.img_sz)
                save_map_to_json(self.parking_map)
        rospy.wait_for_service('get_map')
        rospy.ServiceProxy('get_map', SetBool)(True)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                frame_copy = cv2.warpPerspective(frame.copy(), self.matrix_p, (self.img_sz, self.img_sz))
                frame_copy = cv2.warpAffine(frame_copy, self.matrix_r, (self.img_sz, self.img_sz))
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

                if self.view_contour:
                    for key in self.parking_map['contour'].keys():
                        contour = self.parking_map['contour'][key]
                        cv2.drawContours(frame_copy, [contour], -1, (255, 0, 0), 2)

                if self.view_path:
                    for key in self.parking_map['path'].keys():
                        path = self.parking_map['path'][key]
                        color = (0, 255, 255) if '1' in key else (0, 255, 0)
                        for i in range(path.shape[0] - 1):
                            cv2.line(
                                frame_copy, (int(path[i, 0]), int(path[i, 1])),
                                (int(path[i + 1, 0]), int(path[i + 1, 1])), color, 2, cv2.LINE_AA
                            )

                timestamp = rospy.get_time()
                if ids is not None:
                    for i in range(len(ids)):
                        if int(ids[i]) <= 20:
                            x, y, theta, sz = calc_pose(corners[i].squeeze())
                            cv2.arrowedLine(
                                frame_copy, (int(x), int(y)),
                                (int(x + sz * np.cos(theta)), int(y + sz * np.sin(theta))), (255, 255, 0), 5, 8, 0, 0.25
                            )
                            cv2.putText(
                                frame_copy, str(ids[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 3
                            )
                            self.cars[int(ids[i])] = [x, y, theta, timestamp]
                else:
                    cv2.putText(
                        frame_copy, 'no car detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
                    )

                if self.cars:
                    data = []
                    info = ''
                    for key in list(self.cars.keys()):
                        x, y, theta, t = self.cars[key]
                        if timestamp - t > 2:
                            del self.cars[key]
                        else:
                            data += [key, x, y, theta]
                            info += '\n  car{:>2d}' \
                                    '\n    x:{:>7.2f}, y:{:>7.2f}, theta:{:6.2f}'.format(key, x, y, theta)
                    info = '\n%d car(s) detected' % len(self.cars) + info + '\n' + '-' * 38
                    rospy.loginfo(info)
                    self.pose_pub.publish(Float32MultiArray(data=data))
                else:
                    info = '\nno car detected\n' + '-' * 38
                    rospy.loginfo(info)

                # cv2.imshow('camera_raw', frame)
                cv2.imshow('camera', frame_copy)
                cv2.waitKey(1)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Camera().run()
    except rospy.ROSInterruptException:
        pass
