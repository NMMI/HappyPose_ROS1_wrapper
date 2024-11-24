import rospy
import subprocess
import codecs, json, os
import numpy as np
from pathlib import Path
from happypose_ros.srv import HappyPoseService, HappyPoseServiceRequest, HappyPoseServiceResponse
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, Point, Quaternion

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()

# Path to the dataset folder
main_folder = "/home/alterego-vision/catkin_ws/src/"
dataset_folder = main_folder + "HappyPose Files/"
# Objects map JSON file
objects_map_json = "objects_map.json"
# Output file
output_file = "object_pose.json"
# Camera data file
camera_file = "camera_data.json"
# Image topic
camera_topic = "/camera/color/image_raw_crop"            # Topics from ros_imresize with image at 640x480 pixels
camera_info_topic = "/camera/color/camera_info_crop"     # Topics from ros_imresize with image at 640x480 pixels


# HappyPose Custom pose detector
def camera_info_callback(msg):
    #write camera.json file
    jsonContent = {
        "K": [msg.K[0:3], msg.K[3:6], msg.K[6:9]],
        "resolution": [msg.height, msg.width],
    }

    # Serializing json
    json_object = json.dumps(jsonContent, indent=4)

    # Writing to sample.json
    camera_fn = dataset_folder + camera_file
    with open(camera_fn, "w") as outfile:
        outfile.write(json_object)

    # Unsubscribe after receiving the camera info
    cameraInfoSub.unregister()


def image_callback(msg):
    rospy.loginfo("Image from camera updated!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a png 
        cv2.imwrite(dataset_folder + '/image_rgb.png', cv2_img)


def callback(request):
    rospy.loginfo("Received request for object: %s", request.object)
  
    output_fn = Path(dataset_folder) / output_file    
    if os.path.exists(output_fn):           #Always remove output file prior to inference call if exists
        os.remove(output_fn)

    # Parse objects map JSON to select inference
    data = json.loads(open(dataset_folder + objects_map_json).read())
    if data[request.object] == None:
        rospy.loginfo("Object not found in the dataset")
        exit(-1)

    # Retrieve if to use COSYPOSE or MEGAPOSE
    if data[request.object]['dataset'] == "ycbv":
        detection_type = "--detection-type=cosypose"
        dataset = "--dataset=ycbv"
    else:
        detection_type = "--detection-type=megapose"
        dataset = "--dataset=other"

    mesh_unit = "--mesh-unit=" + data[request.object]['unit']
            

    # Perform your logic here to find the pose
    res = subprocess.run(["python", main_folder + "happypose_ros/scripts/run_inference.py", 
                   data[request.object]['object_name'], dataset_folder, output_file, dataset, detection_type, mesh_unit, "--vis-detections", "--vis-poses"])

    # If detection is good, read the output file and write the estimate

    result = HappyPoseServiceResponse()
    #position = Point(result.position.x, result.position.y, result.position.z)
    #quaternion = Quaternion(result.orientation.x, result.orientation.y, result.orientation.z, result.orientation.w)
    
    if res.returncode == 0 and os.path.exists(output_fn):
        obj_text = codecs.open(output_fn, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        x = np.array(b_new)

        result.success = True
        result.orientation = Quaternion(x[0], x[1], x[2], x[3]) #x,y,z,w
        result.position = Point(x[4], x[5], x[6])

        rospy.loginfo(x)
    else:
        result.success = False

        rospy.loginfo("Object not found")

    return result

def main():
    rospy.init_node('happypose_server')
    rospy.Service('happypose_service', HappyPoseService, callback)
    rospy.Subscriber(camera_topic, Image, image_callback)
    
    # Subscribe to camera info topic and read just first message
    global cameraInfoSub
    cameraInfoSub = rospy.Subscriber(camera_info_topic, CameraInfo, camera_info_callback)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

