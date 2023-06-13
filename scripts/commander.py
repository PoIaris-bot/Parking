#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from digi.xbee.devices import XBeeDevice
from digi.xbee.exception import TransmitException
from std_msgs.msg import Float32MultiArray


class Commander:
    def __init__(self):
        rospy.init_node('commander', anonymous=True)
        # initialize xbee
        node_name = rospy.get_name()
        port = rospy.get_param(node_name + '/port')
        baud_rate = rospy.get_param(node_name + '/baud_rate')
        self.ids = eval(rospy.get_param(node_name + '/ids'))
        self.nodes = {}
        self.xbee_device = XBeeDevice(port, baud_rate)
        self.xbee_device.open()
        xbee_network = self.xbee_device.get_network()
        for node in self.ids:
            self.nodes[node] = xbee_network.discover_device(node)
            rospy.sleep(0.2)

        rospy.Subscriber('/command', Float32MultiArray, self.callback, queue_size=1)
        self.updated = False
        self.data = None

    def callback(self, msg):
        if msg.data:
            self.data = msg.data
            self.updated = True

    def run(self):
        while not rospy.is_shutdown():
            if self.updated:
                for i in range(len(self.data) // 4):
                    node_id = str(int(self.data[i * 4]))
                    angle, direction, speed = map(str, map(int, self.data[i * 4 + 1: i * 4 + 4]))
                    command = '0' * (3 - len(angle)) + angle
                    command += direction
                    command += '0' * (2 - len(speed)) + speed
                    command = 'cmd' + command + 'dmc'
                    # format: cmd[xxx][y][zz]dmc
                    # xxx: angle[75, 105]; y: direction {0: backward, 1: forward}; zz: speed[0, 30]
                    try:
                        self.xbee_device.send_data(self.nodes[node_id], command)
                    except TransmitException:
                        pass
                    except ValueError:
                        pass
                self.updated = False
        # on shutdown: send stop command
        for _ in range(5):
            for node_id in self.nodes.keys():
                try:
                    self.xbee_device.send_data(self.nodes[node_id], 'cmd090000dmc')
                except TransmitException:
                    pass
                except ValueError:
                    pass
                rospy.sleep(0.01)
        self.xbee_device.close()


if __name__ == '__main__':
    try:
        Commander().run()
    except rospy.ROSInterruptException:
        pass
