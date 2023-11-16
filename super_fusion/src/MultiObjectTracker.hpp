// Prevent multiple declarations caused by including
//     this header file from multiple locations
#pragma once

// ROS headers
#include <ros/ros.h>

// Message headers
#include <avs_lecture_msgs/TrackedObjectArray.h>
#include <geometry_msgs/TwistStamped.h>
#include <conti_radar_msgs/RadarObjectArray.h>
#include <yolo_classification/YoloObjectArray.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <super_fusion/MultiObjectTrackerConfig.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/search/kdtree.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include "ObjectEkf.hpp"


// Namespace matches ROS package name
namespace super_fusion {

  class MultiObjectTracker {
    public:
      MultiObjectTracker(ros::NodeHandle& n, ros::NodeHandle& pn);

    private:
      void recvCloud(const sensor_msgs::PointCloud2ConstPtr& msg);
      void updateTimerCallback(const ros::TimerEvent& event);
      void reconfig(MultiObjectTrackerConfig& config, uint32_t level);
      void recvTwist(const geometry_msgs::TwistStamped& msg);
      void recvRadarObjects(const conti_radar_msgs::RadarObjectArrayConstPtr& msg);
      void recvYoloObjects(const yolo_classification::YoloObjectArray& msg);
      void recvCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);

      cv::Rect2d getCamBbox(const avs_lecture_msgs::TrackedObject& object, const tf2::Transform& transform, const image_geometry::PinholeCameraModel& model);

      int getUniqueId();

      conti_radar_msgs::RadarObject applyTransformRadar(const conti_radar_msgs::RadarObject& original_object, geometry_msgs::TransformStamped transform);
      ros::Subscriber sub_twist_, sub_detected_lidar_objects_, sub_detected_radar_objects_, sub_yolo_object_, sub_cam_info_, sub_cloud_;
      ros::Publisher pub_object_tracks_, pub_filtered_radar_, pub_merged_cluster_cloud_, pub_labels_;
      ros::Timer marker_timer_;
      ros::Timer update_timer_;

      dynamic_reconfigure::Server<MultiObjectTrackerConfig> srv_;
      MultiObjectTrackerConfig cfg_;
      geometry_msgs::TwistStamped twist_;
    
      yolo_classification::YoloObjectArray yolo_objects_;
      sensor_msgs::CameraInfo camera_info_;
      image_geometry::PinholeCameraModel model_;
      tf2::Transform camera_transform_; // Coordinate transformation from footprint to camera
      bool looked_up_camera_transform_;
      std::vector<cv::Rect2d> cam_bboxes_;

      tf2_ros::TransformListener listener_;
      tf2_ros::Buffer buffer_;

      std::vector<ObjectEkf> object_ekfs_;
      static constexpr double DT = 1.0;
  };

}
