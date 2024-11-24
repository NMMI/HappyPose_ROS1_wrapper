## HappyPose ROS1 Wrapper
The wrapper takes as input the object name and the 'image_rgb.png' 640x480 image located in HappyPose Files folder.
It looks for the object in all the image and answer with a pose estimation with quaternion and position respect to the camera frame.

If needed, it can take input images from a RealSense camera and publish them at a lower frequency and resized through the ros_imresize node.

[Note]
Happypose within a conda environment must be installed first
Remember to activate conda environment in the same terminal (conda activate happypose) beofre launching the ROS1 wrapper

Before running, in HappyPoseService.py script, line 19 change 'main_folder' with your path to this repository

## Example
Here you can find an example on a sample image with different object from the YCBV dataset.

E.g. to detect the mustard run in a shell the server node with:

rosrun happypose_ros HappyPoseService.py

and then call the service with:
rosservice call happypose_ros/HappyPoseService "object: mustard"

After service answer, you can check pose estimation by output images written in the visualization folder 

If you want to use images from a RealSense camera you can execute the following command:

roslaunch happypose_ros detectObjectFromRealSense.launch

## Support, Bugs and Contribution
Since we are not only focused on this project it might happen that you encounter some trouble once in a while. Maybe we have just forget to think about your specific use case or we have not seen a terrible bug inside our code. In such a case, we are really sorry for the inconvenience and we will provide any support you need.
