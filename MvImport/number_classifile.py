# number_classifier.py

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from auto_aim_interfaces.msg import Armor
from cv_bridge import CvBridge

class NumberClassifier(Node):
    def __init__(self, model_path, label_path, threshold, ignore_classes):
        super().__init__('number_classifier')
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.class_names = self.load_labels(label_path)
        self.threshold = threshold
        self.ignore_classes = ignore_classes
        self.bridge = CvBridge()

        # Publisher for classified armors
        self.armors_pub = self.create_publisher(Armor, '/detector/classified_armors', 10)

        # Subscriber for incoming images
        self.img_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

    def load_labels(self, label_path):
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def image_callback(self, img_msg):
        # Convert ROS image to OpenCV format
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        armors = self.detect_numbers(img)

        for armor in armors:
            self.classify(armor)

    def detect_numbers(self, img):
        # Placeholder for armor detection logic
        # This should return a list of armor objects with number_img attribute
        return []

    def classify(self, armor):
        # Preprocess the image for classification
        number_image = armor.number_img
        number_image = cv2.cvtColor(number_image, cv2.COLOR_BGR2GRAY)
        number_image = cv2.threshold(number_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        number_image = number_image / 255.0  # Normalize

        # Create blob from image
        blob = cv2.dnn.blobFromImage(number_image)

        # Set the input blob for the neural network
        self.model.setInput(blob)

        # Forward pass the image blob through the model
        outputs = self.model.forward()

        # Softmax to get probabilities
        softmax_prob = np.exp(outputs - np.max(outputs))
        softmax_prob /= np.sum(softmax_prob)

        # Get the class with the highest probability
        class_id = np.argmax(softmax_prob)
        confidence = softmax_prob[class_id]

        if confidence >= self.threshold and self.class_names[class_id] not in self.ignore_classes:
            armor.confidence = confidence
            armor.number = self.class_names[class_id]
            self.armors_pub.publish(armor)

def main(args=None):
    rclpy.init(args=args)
    model_path = 'path/to/your/model.onnx'
    label_path = 'path/to/your/labels.txt'
    threshold = 0.7
    ignore_classes = ['negative']

    classifier = NumberClassifier(model_path, label_path, threshold, ignore_classes)
    rclpy.spin(classifier)
    classifier.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()