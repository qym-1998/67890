import rospy
import numpy as np

from hector_uav_msgs.srv import EnableMotors
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

import tf.transformations as ttt

import cv2
from cv_bridge import CvBridge, CvBridgeError

cv_bridge_inst = CvBridge()

ALTITUDE_DESIRED_VALUE = 5.0
kz = 1.0
bz = 2.0

ks = 0.01
bs = 0.005

ky = 0.008
by = 0.00005


RING_AVOIDANCE_TIME = 5  # [seconds]
DEFAULT_ALTITUDE = 3  # [meters]

class Controller:
    def __init__(self):
        rospy.init_node("controller_node")
        rospy.Subscriber("/ground_truth/state", Odometry, self.state_callback)
        rospy.Subscriber("/cam_1/camera/image", Image, self.camera1_callback)
        rospy.Subscriber("/cam_2/camera/image", Image, self.camera2_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.position = Point()
        self.twist = Twist()
        self.omega_error = 0
        self.omega_vel = 0
        self.omega_error_prev = 0
        self.y_error = 0
        self.y_error_prev = 0
        self.z_des = DEFAULT_ALTITUDE
        self.image_1 = NONE
        self.image_2 = NONE
        self.e_x_blue = 0
        self.e_y_blue = 0
        self.red_ring_detected = False
        self.blue_ring_detected = False
        self.state = "free_flight"
        self.time_start_up = 0
        self.avoidance_time = 0
        self.cv_bridge = CvBridge()

    def __del__(self):
        self.stop_robot()

    def stop_robot(self):
        
     #if self.cmd_pub.get_num_connections() > 0:
        cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        cmd_msg = Twist()
        cmd_msg.linear.x = 0
        cmd_msg.linear.y = 0
        cmd_msg.linear.z = 0
        cmd_msg.angular.z = 0
        cmd_pub.publish(cmd_msg)

        self.cmd_pub.publish(cmd_msg)

    #def obom_callback(self, msg):
        """ Pose of a robot extraction"""
        transform = self.tfBuffer.lookup_transform('world', 'base_stabilized', rospy.Time()).transform
        x, y, z = transform.translation.x, transform.translation.y, transform.translation.z
        quat = transform.rotation
        r, p, y = ttt.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        self.drone_state = [x, y, z, r, p, y]
    

    def state_callback(self, msg):
        self.position = msg.pose.pose.position
        self.twist = msg.twist.twist

    def camera1_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        grey_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey_image, 8, 255, cv2.THRESH_BINARY_INV)
        cv_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.line(cv_image, (160, 0), (160, 240), (0, 123, 0), 1)
        cv2.line(cv_image, (0, 120), (320, 120), (0, 123, 0), 1)
        top_points = np.where(mask[10] >= 10)
        mid_points = np.where(mask[msg.height // 2] >= 10)
        if len(top_points[0]) > 0 and len(mid_points[0]) > 0:
            top_line_point = int(np.average(top_points))
            mid_line_point = int(np.average(mid_points))
            self.omega_error = top_line_point - mid_line_point
        else:
            self.omega_error = 0
        __, cy_list = np.where(mask >= 10)
        if len(cy_list) > 0:
            cy = int(np.average(cy_list))
            self.y_error = msg.width / 2 - cy
        # "steering" conrol
        top_points = np.where(mask[10] >= 10)
        mid_points = np.where(mask[msg.height / 2] >= 10)
        if  (not np.isnan(np.average(top_points)) and not np.isnan(np.average(mid_points))):
            top_line_point = int(np.average(top_points))
            mid_line_point = int(np.average(mid_points))
            self.omega_error = top_line_point - mid_line_point
            
            cv2.circle(cv_image, (top_line_point, 10), 5, (0,0,255), 1)
            cv2.circle(cv_image, (mid_line_point, int(msg.height/2)), 5, (0,0,255), 1)
            cv2.line(cv_image, (mid_line_point, int(msg.height/2)), (top_line_point, 10), (0, 0, 255), 3)

      # y-offset control
        __, cy_list = np.where(mask >= 10)
        if not np.isnan(np.average(cy_list)):
            cy = int(np.average(cy_list))
            self.y_error = msg.width / 2 - cy
            
            cv2.circle(cv_image, (cy, int(msg.height/2)), 7, (0,255,0), 1)
            cv2.line(cv_image, (160, 120), (cy, int(msg.height/2)), (0, 255, 0), 3)

        self.show_image_1(cv_image)
        #self.image_1 = cv_image

    def camera2_callback(self, msg):
         # """ Computer vision stuff for Rings"""
       try:
           cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
       except CvBridgeError as e:
           rospy.logerr("CvBridge Error: {0}".format(e))


       # red
       lower = np.uint8([0, 0, 90])
       upper = np.uint8([30, 30, 120])
       cv_image, red_pose, red_radius  = self.ring_detector(cv_image, lower, upper, (0,0,255))


       # blue
       lower = np.uint8([40, 20, 20])
       upper = np.uint8([80, 50, 50])
       cv_image, blue_pose, blue_radius = self.ring_detector(cv_image, lower, upper, (255,0,0))


       # print(red_radius, blue_radius)


       if 50 < red_radius < 70 or 50 < blue_radius < 80:
           if red_radius > blue_radius:
               self.blue_ring_detected = False
               self.red_ring_detected = True
           else:
               self.red_ring_detected = False
               self.blue_ring_detected = True
              
               # offset in ring xy-plane to fly through center of a ring
               # error = <center of image> - <center of ring>
               self.e_x_blue = 160 - blue_pose[0]
               self.e_y_blue = 120 - blue_pose[1]
       else:
           self.blue_ring_detected = False
           self.red_ring_detected = False


       # save results
       #self.show_image_2(cv_image)
       self.image_2 = cv_image

    def show_image_1(self, img, title='Camera 1'):
        cv2.imshow(title, img)
        cv2.waitKey(3)
    
    def show_image_2(self, img, title='Camera 2'):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def ring_detector(self, image, lower, upper, color):
        color_mask = cv2.inRange(image, lower, upper)
        color_contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if color_contours:
            max_len_c = 0
            c = color_contours[0]
            for i in range(0, len(color_contours)):
                if len(color_contours[i]) > max_len_c:
                    c = color_contours[i]
                    max_len_c = len(color_contours[i])
            self.color_distance = max_len_c
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx = 0
                cy = 0
            (x1, y1), color_r = cv2.minEnclosingCircle(c)
            if color_r > 10:
                image = cv2.circle(image, (cx, cy), radius=5, color=color, thickness=-1)
                cv2.drawContours(color_r, c, -1, (0, 255, 0), 1)
                color_r = cv2.circle(color_r, (int(x1), int(y1)), radius=int(color_r), color=color, thickness=4)
                return image, (x1, y1), color_r[0]
        return image, (0, 0), 0

    def fsm_update(self):
        if self.red_ring_detected:
            self.state = "drone_up"
        elif RING_AVOIDANCE_TIME < self.avoidance_time < RING_AVOIDANCE_TIME + 4:
            self.state = "drone_down"
            self.time_start_up = 0
        elif self.blue_ring_detected:
            self.state = "drone_blue_ring"
        else:
            self.state = "free_flight"

    def enable_motors(self):
        rospy.wait_for_service("/enable_motors")
        foo2call = rospy.ServiceProxy("/enable_motors", EnableMotors)
        if foo2call(True):
          print("Motors started!")
    
    def takeoff_ctrl(self, z_des):

      return  kz * (ALTITUDE_DESIRED_VALUE - self.position.z) - bz * self.twist.linear.z
  
    def steering_strl(self):
      u_s =  ks * self.omega_error - bs * (self.omega_error - self.omega_error_prev) / (1.0 / 50.0)
      self.omega_error_prev = self.omega_error
      return u_s


    def y_offset_ctrl(self):
      u_y = ky * self.y_error - by * (self.y_error - self.y_error_prev) / (1.0 / 50.0)
      self.y_error_prev = self.y_error
      return u_y


    def spin(self):
        self.enable_motors()
        try:
            rate = rospy.Rate(50.0)
            while not rospy.is_shutdown():
                self.fsm_update()
                if self.state == "drone_up":
                    self.z_des = 5
                    if self.time_start_up == 0:
                        self.time_start_up = rospy.get_time()
                elif self.state == "drone_down":
                    self.z_des = DEFAULT_ALTITUDE
                elif self.state == "drone_blue_ring":
                    self.z_des += 0.001 * self.e_y_blue
                elif self.state == "free_flight":
                    pass
                else:
                    rospy.logerr("Error: state name error!")

                self.avoidance_time = rospy.get_time() - self.time_start_up
                print(self.state, self.z_des)

                u_z = self.takeoff_ctrl(DEFAULT_ALTITUDE)
                u_s = self.steering_strl()
                u_y = self.y_offset_ctrl()

                cmd_msg = Twist()
                cmd_msg.linear.z = u_z
                cmd_msg.linear.x = 2.2
                cmd_msg.linear.y = u_y
                cmd_msg.angular.z = -u_s

                self.cmd_pub.publish(cmd_msg)
                rate.sleep()
        except:
            self.stop_robot()
            print("stop the robot!")



def main():
    ctrl = Controller()
    ctrl.spin()

if __name__ == "__main__":
    main()