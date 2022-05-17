import mediapipe as mp
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from geometry_msgs.msg import Twist
  
class ImagePublisher(Node):
  """
  Create an ImagePublisher class, which is a subclass of the Node class.
  """
  def __init__(self):
    super().__init__('image_publisher')
    pub_cmd_vel = self.create_publisher(Twist,"cmd_vel", 10)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands          
    cap = cv2.VideoCapture(2)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands: 
      while cap.isOpened():
          ret, frame = cap.read()
          
          # BGR 2 RGB
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
          # Flip on horizontal
          image = cv2.flip(image, 1)
          
          # Set flag
          image.flags.writeable = False
          
          # Detections
          results = hands.process(image)
          
          # Set flag to true
          image.flags.writeable = True
          
          # RGB 2 BGR
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          
          # Rendering results
          if results.multi_hand_landmarks:
              hand = results.multi_hand_landmarks[0]
              mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )
              try:
                landmarks = hand.landmark
                mcp = mp_hands.HandLandmark.THUMB_MCP
                tip = mp_hands.HandLandmark.THUMB_TIP
                thumb_mcp = [landmarks[mcp.value].x,landmarks[mcp.value].y]
                thumb_tip = [landmarks[tip.value].x,landmarks[tip.value].y]
                
                diff_x = thumb_mcp[0] - thumb_tip[0]
                diff_y = thumb_mcp[1] - thumb_tip[1]

                if abs(diff_x) > abs(diff_y):
                  if diff_x > 0:
                    print("Left")
                    vel = Twist()
                    vel.angular.z= 0.1 
                    pub_cmd_vel.publish(vel)
                  else:
                    print("Right")
                    vel = Twist()
                    vel.angular.z= -0.1 
                    pub_cmd_vel.publish(vel)
                else:
                  if diff_y > 0:
                    print("Front")
                    vel = Twist()
                    vel.linear.x= 0.1 
                    pub_cmd_vel.publish(vel)
                  else:
                    print("Back")
                    vel = Twist()
                    vel.linear.x= -0.1 
                    pub_cmd_vel.publish(vel)
              except Exception as err:
                  print(err)
                  pass
              
          
          cv2.imshow('Hand Tracking', image)

          if cv2.waitKey(10) & 0xFF == ord('q'):
              break
    cap.release()
    cv2.destroyAllWindows()  
  
   
def main(args=None):
   
  # Initialize the rclpy library
  rclpy.init(args=args)
   
  # Create the node
  image_publisher = ImagePublisher()
   
  # Spin the node so the callback function is called.
  rclpy.spin(image_publisher)
   
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_publisher.destroy_node()
   
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()

