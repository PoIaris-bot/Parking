#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import rospy
import numpy as np
import cv2.aruco as aruco
from std_msgs.msg import Float32MultiArray
from tools import calc_line_cross_point, calc_pose


class Camera:
    def __init__(self):
        rospy.init_node('camera', anonymous=True)
        self.pose_pub = rospy.Publisher('/pose', Float32MultiArray, queue_size=1)

        node_name = rospy.get_name()
        cam = rospy.get_param(node_name + '/cam')

        self.img_sz = (640, 480)
        self.cap = cv2.VideoCapture(cam, cv2.WINDOW_NORMAL)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_sz[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_sz[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.area_sz = 500
        self.matrix_p, self.matrix_r, self.contours = self.preprocess()

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # 设置预定义的字典
        self.aruco_param = aruco.DetectorParameters_create()
        self.markers = {}

        rospy.loginfo('camera initialization finished')
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

                area = np.zeros_like(gray, dtype=np.uint8)
                count = 0
                area_center = []
                area_contours = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if x not in [0, self.img_sz[0] - w] and y not in [0, self.img_sz[1] - h]:
                        area_contours.append(contour)
                        m = cv2.moments(contour)
                        xc = int(m['m10'] / m['m00'])
                        yc = int(m['m01'] / m['m00'])
                        area_center.append([xc, yc])
                        count += 1
                    if count == 2:
                        break
                convex_hull = cv2.convexHull(np.vstack(area_contours))
                cv2.drawContours(area, [convex_hull], -1, 255, cv2.FILLED)

                fld = cv2.ximgproc.createFastLineDetector()
                lines = fld.detect(area).squeeze().tolist()
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
                    np.float32([[0, 0], [self.area_sz, 0], [0, self.area_sz], [self.area_sz, self.area_sz]])
                )

                angle_dict = {1: 270, 2: 180, -1: 0, -2: 90}
                angle = np.arctan2(area_center[0][1] - area_center[1][1], area_center[1][0] - area_center[0][0])
                quadrant = int(angle / np.pi * 2) + (1 if angle > 0 else -1)

                matrix_r = cv2.getRotationMatrix2D(
                    (self.area_sz // 2, self.area_sz // 2), angle_dict[quadrant], 1
                )

                image = cv2.warpPerspective(frame, matrix_p, (self.area_sz, self.area_sz))
                image = cv2.warpAffine(image, matrix_r, (self.area_sz, self.area_sz))

                _, binary = cv2.threshold(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
                return matrix_p, matrix_r, contours[:4]

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                frame_copy = cv2.warpPerspective(frame.copy(), self.matrix_p, (self.area_sz, self.area_sz))
                frame_copy = cv2.warpAffine(frame_copy, self.matrix_r, (self.area_sz, self.area_sz))
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_param)
                cv2.drawContours(frame_copy, self.contours, -1, (255, 0, 0), 2)

                timestamp = rospy.get_time()
                if ids is not None:
                    for i in range(len(ids)):
                        x, y, theta, sz = calc_pose(corners[i].squeeze())
                        cv2.arrowedLine(
                            frame_copy, (int(x), int(y)), (int(x + sz * np.cos(theta)), int(y + sz * np.sin(theta))),
                            (0, 0, 255), 5, 8, 0, sz / 100
                        )
                        cv2.putText(
                            frame_copy, str(ids[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        self.markers[int(ids[i])] = [x, y, theta, timestamp]
                else:
                    cv2.putText(
                        frame_copy, 'no marker detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                    )

                if self.markers:
                    data = []
                    info = ''
                    for key in list(self.markers.keys()):
                        x, y, theta, t = self.markers[key]
                        if t < timestamp:
                            del self.markers[key]
                        else:
                            data += [key, x, y, theta]
                            info += '\n  marker{:>2d}'.format(key)
                            info += '\n    x:{:>7.2f}, y:{:>7.2f}, theta:{:6.2f}'.format(x, y, theta)
                    info = '\n%d marker(s) detected' % len(self.markers) + info + '\n' + '-' * 38
                    rospy.loginfo(info)
                    self.pose_pub.publish(Float32MultiArray(data=data))
                else:
                    info = '\nno marker detected\n' + '-' * 38
                    rospy.loginfo(info)

                # cv2.imshow('camera_raw', frame)
                cv2.imshow('camera', frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Camera()
    except rospy.ROSInterruptException:
        pass
