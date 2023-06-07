#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from pathlib import Path
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, SetBoolResponse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class Planner:
    def __init__(self):
        rospy.init_node('planner', anonymous=True)

        self.cars = {}
        rospy.Subscriber('/pose', Float32MultiArray, self.callback, queue_size=1)

        self.lot = None
        self.lane = None
        self.spot1 = None
        self.spot2 = None
        rospy.Service('get_map', SetBool, self.handler)

        rospy.spin()

    def callback(self, msg):
        data = msg.data
        if self.lot is not None and data:
            for i in range(len(data) // 4):
                self.cars[int(data[i * 4])] = data[i * 4 + 1: i * 4 + 4]
            print(self.cars)

    def handler(self, req):
        if req.data:
            try:
                self.lot = np.loadtxt(ROOT / 'map/lot.txt', 'int')
                self.lane = np.loadtxt(ROOT / 'map/lane.txt', 'int')
                self.spot1 = np.loadtxt(ROOT / 'map/spot1.txt', 'int')
                self.spot2 = np.loadtxt(ROOT / 'map/spot2.txt', 'int')
            except FileNotFoundError:
                pass

            if self.lot is not None:
                return SetBoolResponse(success=True)
            else:
                return SetBoolResponse(success=False)
        else:
            return SetBoolResponse(success=False)


if __name__ == '__main__':
    try:
        Planner()
    except rospy.ROSInterruptException:
        pass
