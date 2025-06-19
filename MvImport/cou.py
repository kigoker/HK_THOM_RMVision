import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    global bridge
    try:
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imshow("frame", cv_img)
        cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

if __name__ == '__main__':
    rospy.init_node('image_processing_node', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber('image_out', Image, callback)
    rospy.spin()