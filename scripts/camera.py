#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import rospy
import numpy as np
import cv2.aruco as aruco
from std_srvs.srv import SetBool
from std_msgs.msg import Float32MultiArray
from tools import calc_pose, preprocess, save_map_to_json


class Camera:
    def __init__(self):
        rospy.init_node('camera', anonymous=True)
        # get ros parameters
        node_name = rospy.get_name()
        cam = rospy.get_param(node_name + '/cam')
        self.img_sz = rospy.get_param(node_name + '/img_sz')
        self.view_raw = rospy.get_param(node_name + '/view_raw')
        self.view_path = rospy.get_param(node_name + '/view_path')
        self.view_contour = rospy.get_param(node_name + '/view_contour')
        self.ids = list(map(int, eval(rospy.get_param(node_name + '/ids'))))

        self.cap = cv2.VideoCapture(cam)  # open camera
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # set frame per second

        # initialize ArUco
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        self.cars = {}
        self.pose_pub = rospy.Publisher('/pose', Float32MultiArray, queue_size=1)

        # preprocess: perspective and rotation transform, calculate map contours and paths
        ret = False
        while not ret:
            ret, frame = self.cap.read()
            if ret:
                self.matrix_p, self.matrix_r, self.parking_map = preprocess(frame, self.img_sz)
                save_map_to_json(self.parking_map)  # save map
        # ask node 'controller' to access map
        rospy.wait_for_service('get_map')
        get_map = rospy.ServiceProxy('get_map', SetBool)
        while not get_map(True):
            pass

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                frame_copy = cv2.warpPerspective(frame.copy(), self.matrix_p, (self.img_sz, self.img_sz))
                frame_copy = cv2.warpAffine(frame_copy, self.matrix_r, (self.img_sz, self.img_sz))
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

                if self.view_contour:  # draw contours
                    for key in self.parking_map['contour'].keys():
                        contour = self.parking_map['contour'][key]
                        cv2.drawContours(frame_copy, [contour], -1, (255, 0, 0), 2)

                if self.view_path:  # draw paths
                    for key in self.parking_map['path'].keys():
                        path = self.parking_map['path'][key]
                        color = (0, 255, 255) if '1' in key else ((0, 255, 0) if '2' in key else (255, 255, 0))
                        for i in range(path.shape[0] - 1):
                            cv2.line(
                                frame_copy, (int(path[i, 0]), int(path[i, 1])),
                                (int(path[i + 1, 0]), int(path[i + 1, 1])), color, 2, cv2.LINE_AA
                            )

                if ids is not None:  # draw direction arrows and ids
                    for i in range(len(ids)):
                        if int(ids[i]) in self.ids:
                            x, y, theta, sz = calc_pose(corners[i].squeeze())
                            cv2.arrowedLine(
                                frame_copy, (int(x), int(y)),
                                (int(x + sz * np.cos(theta)), int(y + sz * np.sin(theta))), (0, 0, 255), 5, 8, 0, 0.25
                            )
                            cv2.putText(
                                frame_copy, str(ids[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (255, 255, 0), 2
                            )
                            self.cars[int(ids[i])] = [x, y, theta]
                else:
                    cv2.putText(
                        frame_copy, 'no car detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
                    )

                if self.cars:  # publish pose topic
                    data = []
                    for key in list(self.cars.keys()):
                        x, y, theta = self.cars[key]
                        data += [key, x, y, theta]
                    self.pose_pub.publish(Float32MultiArray(data=data))

                if self.view_raw:
                    cv2.imshow('camera_raw', frame)
                cv2.imshow('camera', frame_copy)
                cv2.waitKey(1)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Camera().run()
    except rospy.ROSInterruptException:
        pass
