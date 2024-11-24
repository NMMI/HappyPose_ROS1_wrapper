#include "ros_imresize/image_handler.h"

#include <ros/ros.h>

int main( int argc, char** argv )
{
    ros::init(argc, argv, "ros_imresize");

    SingleImageHandler resize;

    ros::spin();

    return 0;
}
