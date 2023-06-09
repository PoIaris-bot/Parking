#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import rospy
import numpy as np
from pathlib import Path
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, SetBoolResponse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def load_data_from_json():
    with open(ROOT / 'map.json', 'r') as file:
        parking_map = json.load(file)
    for key1 in parking_map.keys():
        for key2 in parking_map[key1].keys():
            parking_map[key1][key2] = np.array(parking_map[key1][key2])

    return parking_map


class Planner:
    def __init__(self):
        rospy.init_node('planner', anonymous=True)

        self.cars = {}
        rospy.Subscriber('/pose', Float32MultiArray, self.callback, queue_size=1)

        self.parking_map = None
        rospy.Service('get_map', SetBool, self.handler)

        rospy.spin()

    def callback(self, msg):
        data = msg.data
        if self.parking_map is not None and data:
            for i in range(len(data) // 4):
                self.cars[int(data[i * 4])] = data[i * 4 + 1: i * 4 + 4]
            print(self.cars)

    def handler(self, req):
        if req.data:
            try:
                self.parking_map = load_data_from_json()
            except FileNotFoundError:
                pass

            if self.parking_map is not None:
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
