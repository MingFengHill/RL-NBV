import logging
import roslib
import rospy
import tf
import time
from threading import Thread, Lock
from std_msgs.msg import Int32
import numpy as np


class InformationGainAdapter:
    def __init__(self):
        rospy.init_node("information_gain_adapter")
        self.camera_id_sub = rospy.Subscriber("camera_id_inner", Int32, self.camera_id_callback)
        self.camera_id_pub = rospy.Publisher('camera_id', Int32, queue_size=10)
        self.br = tf.TransformBroadcaster()
        self.tf_thread = Thread(target=self.send_camera_transform)
        self.camera_poses = np.loadtxt("./poses.txt")
        self.camera_id = None
        self.camera_id_lock = Lock()
        self.start_tf_thr()
        self.model_cnt = 1
        # camera
        self.camera_id_update_pub = rospy.Publisher('camera_id_update', Int32, queue_size=10)

    def start_tf_thr(self):
        self.is_run = True
        self.tf_thread.start()

    def stop_tf_thr(self):
        self.is_run = False
        self.tf_thread.join()

    def send_camera_transform(self):
        rate = rospy.Rate(10.0)
        while self.is_run:
            self.camera_id_lock.acquire()
            cur_id = self.camera_id
            self.camera_id_lock.release()
            if cur_id == None:
                continue
            camera_pose = self.camera_poses[cur_id]
            self.br.sendTransform((camera_pose[0], camera_pose[1], camera_pose[2]),
                                  (camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]),
                                  rospy.Time.now(),
                                  "camera",
                                  "world")
            self.camera_id_pub.publish(cur_id)
            # for i in range(33):
            #     camera_pose = self.camera_poses[i]
            #     cur_frame = "cam_pos_{}".format(i)
            #     self.br.sendTransform((camera_pose[0], camera_pose[1], camera_pose[2]),
            #                           (camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]),
            #                           rospy.Time.now(),
            #                           cur_frame,
            #                           "world")
            rate.sleep()

    def camera_id_callback(self, data):
        self.camera_id_lock.acquire()
        if data.data == 999:
            self.camera_id = None
            print("[INFO] stop camara id pub")
            print("[INFO] current model cnt: {}".format(self.model_cnt))
            self.model_cnt += 1
        elif data.data >= 0 and data.data < 33:
            self.camera_id = data.data
            print("[INFO] currrent camera id: {}".format(self.camera_id))
        else:
            print("[ERROR] camera id: {}".format(data.data))
        self.camera_id_lock.release()
        self.camera_id_update_pub.publish(1)


# for debugging purposes
class DummyPublisher:
    def __init__(self):
        # rospy.init_node("dummy_publisher")
        self.camera_id_pub = rospy.Publisher('camera_id', Int32, queue_size=10)

    def pub_camera_id(self, camera_id):
        self.camera_id_pub.publish(camera_id)


if __name__ == '__main__':
    adapter = InformationGainAdapter()
    while True:
        option = input("Have fun~ :)\n1. Publish Camera ID;\n"
                       "2. Exit.\nInput:")
        if option == 1:
            pass
        elif option == 2:
            print("bye bye~")
            adapter.stop_tf_thr()
            break
        else:
            print("[WARR] Wrong Input {}".format(option))
    
    # dummy_pub = DummyPublisher()
    # while True:
    #     option = input("Have fun~ :)\n1. Publish Camera ID;\n"
    #                    "2. Exit.\nInput:")
    #     if option == 1:
    #         camera_id = input("Please Input Camera ID:")
    #         dummy_pub.pub_camera_id(camera_id)
    #     elif option == 2:
    #         print("bye bye~")
    #         adapter.stop_tf_thr()
    #         break
    #     else:
    #         print("[WARR] Wrong Input {}".format(option))
    