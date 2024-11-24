#include "ros_imresize/image_handler.h"

#include <cv_bridge/cv_bridge.h>
#include <ros/callback_queue.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>

using namespace std;

SingleImageHandler::SingleImageHandler() :
_nh("/ros_imresize"),
_width(0),
_height(0),
_it(_nh)
{
    string imgTopicName;
    string infoTopicName;
    bool set_img = false;
    float resizeFrequency = 10;

    ros::param::get("resize_width", _width);
    ros::param::get("resize_height", _height);
    ros::param::get("topic_crop", imgTopicName);
    ros::param::get("camera_info", infoTopicName);
    ros::param::get("resize_frequency", resizeFrequency);

    ros::Subscriber sub_info = _nh.subscribe(infoTopicName, 1, &SingleImageHandler::setCameraInfo, this);

    _sub_img = _it.subscribe(imgTopicName, 1, &SingleImageHandler::topicCallback, this);

    _pub_img = _it.advertise(imgTopicName + "_crop", 1);

    _pub_info = _nh.advertise<sensor_msgs::CameraInfo>(infoTopicName + "_crop", 1);
    
    ros::Rate wrait(resizeFrequency);

    ROS_INFO("Waiting for camera topics ...");

    while(ros::ok())
    {
	ros::spinOnce();
	wrait.sleep();
    }
}

SingleImageHandler::~SingleImageHandler()
{
    ROS_INFO("Shutdown node");
    ros::shutdown();
}

void SingleImageHandler::topicCallback(const sensor_msgs::ImageConstPtr& received_image)
{
    cv_bridge::CvImagePtr cvPtr;
    cvPtr = cv_bridge::toCvCopy(received_image, sensor_msgs::image_encodings::RGB8);
       
    cv::Mat undist;
    undist = cvPtr->image;
    cv::resize(undist, cvPtr->image, cv::Size(_width, _height),
               0, 0, cv::INTER_LINEAR);

    _pub_img.publish(cvPtr->toImageMsg());
    _pub_info.publish(_infoCam);

    if (!set_img)
    {
        ofstream fd;
        fd.open("/home/alterego-vision/catkin_ws_pose/src/visp_megapose/params/camera.txt", ios::out);   
        fd <<setprecision(10)<<_infoCam.K[0]<<endl<<_infoCam.K[2]<<endl<<_infoCam.K[4]<<endl<<_infoCam.K[5]; /* scrittura dati */
        fd.close();
        set_img = true;
        ROS_INFO("Print new camera calibration on camera.txt file!");
        ROS_INFO("Resize node is running for image topic!");
    }
}

void SingleImageHandler::setCameraInfo(const sensor_msgs::CameraInfoConstPtr &received_info)
{
    _infoCam = *received_info;

    float scale_x = (float)(_width) / (float)(_infoCam.width);
    float scale_y = (float)(_height) / (float)(_infoCam.height);

    _infoCam.K[0] *= scale_x;
    _infoCam.K[2] *= scale_x;

    _infoCam.K[4] *= scale_y;
    _infoCam.K[5] *= scale_y;
    
    _infoCam.P[0] *= scale_x;
    _infoCam.P[2] *= scale_x;

    _infoCam.P[5] *= scale_y;
    _infoCam.P[6] *= scale_y;

    _infoCam.width = _width;
    _infoCam.height = _height;

    // ROS_INFO_STREAM("Previous camera info :\n" << *received_info << "\n");
    // ROS_INFO_STREAM("New camera info :\n" << _infoCam << "\n");
    ROS_INFO_STREAM_ONCE("Resize node is running for camera info topic!");
}