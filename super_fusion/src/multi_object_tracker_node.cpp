// ROS and node class header file
#include <ros/ros.h>
#include "MultiObjectTracker.hpp"

int main(int argc, char** argv)
{
  // Initialize ROS and declare node handles
  ros::init(argc, argv, "multi_object_tracker");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");
  
  // Instantiate node class
  super_fusion::MultiObjectTracker node(n, pn);

  // Spin and process callbacks
  ros::spin();
}
