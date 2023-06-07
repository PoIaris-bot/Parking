#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import rospy
import numpy as np
import cv2.aruco as aruco
from pathlib import Path
from std_srvs.srv import SetBool
from std_msgs.msg import Float32MultiArray
from tools import calc_line_cross_point, calc_pose

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class Camera:
    def __init__(self):
        rospy.init_node('camera', anonymous=True)

        self.pose_pub = rospy.Publisher('/pose', Float32MultiArray, queue_size=1)

        node_name = rospy.get_name()
        cam = rospy.get_param(node_name + '/cam')
        self.cap = cv2.VideoCapture(cam, cv2.WINDOW_NORMAL)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.raw_sz = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.img_sz = rospy.get_param(node_name + '/img_sz')

        self.matrix_p, self.matrix_r, self.lot, self.lane, self.spot1, self.spot2 = self.preprocess()
        rospy.wait_for_service('get_map')
        rospy.ServiceProxy('get_map', SetBool)(True)

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()
        self.cars = {}

        self.run()

    def preprocess(self):
        ret = False
        while not ret:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)

                count = 0
                area_centers = []
                area_contours = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if x not in [0, self.raw_sz[0] - w] and y not in [0, self.raw_sz[1] - h]:
                        area_contours.append(contour)
                        m = cv2.moments(contour)
                        xc = int(m['m10'] / m['m00'])
                        yc = int(m['m01'] / m['m00'])
                        area_centers.append([xc, yc])
                        count += 1
                    if count == 2:
                        break
                convex_hull = np.zeros_like(gray, dtype=np.uint8)
                cv2.drawContours(convex_hull, [cv2.convexHull(np.vstack(area_contours))], -1, 255, cv2.FILLED)

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
                    np.float32([[0, 0], [self.img_sz, 0], [0, self.img_sz], [self.img_sz, self.img_sz]])
                )

                angle_dict = {1: 270, 2: 180, -1: 0, -2: 90}
                angle = np.arctan2(area_centers[0][1] - area_centers[1][1], area_centers[1][0] - area_centers[0][0])
                quadrant = int(angle / np.pi * 2) + (1 if angle > 0 else -1)

                matrix_r = cv2.getRotationMatrix2D(
                    (self.img_sz // 2, self.img_sz // 2), angle_dict[quadrant], 1
                )

                image = cv2.warpPerspective(frame, matrix_p, (self.img_sz, self.img_sz))
                image = cv2.warpAffine(image, matrix_r, (self.img_sz, self.img_sz))

                _, image = cv2.threshold(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                area_contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                area_contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)

                area_contours = [contour.squeeze() for contour in area_contours]
                if area_contours[2][0, 0] > area_contours[3][0, 0]:
                    area_contours[2], area_contours[3] = area_contours[3], area_contours[2]
                lot, lane, spot1, spot2 = area_contours[:4]

                if not os.path.exists(ROOT / 'map'):
                    os.makedirs(ROOT / 'map')
                np.savetxt(ROOT / 'map/lot.txt', lot, '%d')
                np.savetxt(ROOT / 'map/lane.txt', lane, '%d')
                np.savetxt(ROOT / 'map/spot1.txt', spot1, '%d')
                np.savetxt(ROOT / 'map/spot2.txt', spot2, '%d')
                return matrix_p, matrix_r, lot, lane, spot1, spot2

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                frame_copy = cv2.warpPerspective(frame.copy(), self.matrix_p, (self.img_sz, self.img_sz))
                frame_copy = cv2.warpAffine(frame_copy, self.matrix_r, (self.img_sz, self.img_sz))
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                cv2.drawContours(frame_copy, [self.lot, self.lane, self.spot1, self.spot2], -1, (255, 0, 0), 2)

                timestamp = rospy.get_time()
                if ids is not None:
                    for i in range(len(ids)):
                        x, y, theta, sz = calc_pose(corners[i].squeeze())
                        cv2.arrowedLine(
                            frame_copy, (int(x), int(y)), (int(x + sz * np.cos(theta)), int(y + sz * np.sin(theta))),
                            (0, 0, 255), 5, 8, 0, 0.25
                        )
                        self.cars[int(ids[i])] = [x, y, theta, timestamp]
                    aruco.drawDetectedMarkers(frame_copy, corners, ids, borderColor=(255, 0, 0))
                else:
                    cv2.putText(
                        frame_copy, 'no car detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows()

        os.system('rosnode kill -a')
        pid = os.popen('pgrep rosmaster').read().strip()
        os.system('kill -9 %s' % pid)


if __name__ == '__main__':
    try:
        Camera()
    except rospy.ROSInterruptException:
        pass
