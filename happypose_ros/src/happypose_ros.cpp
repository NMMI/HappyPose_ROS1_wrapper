#include <ros/ros.h>
#include <happypose_ros/HappyPoseService.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "happypose_client");
  ros::NodeHandle n;

  // Get param from launch file
  std::string objName = "";
  n.getParam("objectName", objName);

  // Wait for the service to be available
  ros::ServiceClient client;
  client = n.serviceClient<happypose_ros::HappyPoseService>("happypose_service");

  // Send a request to service
  happypose_ros::HappyPoseService srv;
  srv.request.object = objName;

  if (client.call(srv)) {
    ROS_INFO("Response from service: %s", srv.response.success ? "true" : "false");
    ROS_INFO("Quaternion: x=%f, y=%f, z=%f, w=%f", srv.response.orientation.x, srv.response.orientation.y, srv.response.orientation.z, srv.response.orientation.w);
    ROS_INFO("Position: x=%f, y=%f, z=%f", srv.response.position.x, srv.response.position.y, srv.response.position.z);
  } else {
    ROS_ERROR("Call to service failed");
    return 1;
  }

  return 0;
}

