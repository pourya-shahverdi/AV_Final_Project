// Header file for the class
#include "MultiObjectTracker.hpp"
#include "LidarFilter.hpp"

// Namespace matches ROS package name
namespace super_fusion
{

  // Constructor with global and private node handle arguments
  MultiObjectTracker::MultiObjectTracker(ros::NodeHandle &n, ros::NodeHandle &pn) : listener_(buffer_)
  {
    // recv radar and lidar
    pub_object_tracks_ = n.advertise<avs_lecture_msgs::TrackedObjectArray>("multi_object_tracker/object_tracks", 1);
    pub_filtered_radar_ = n.advertise<avs_lecture_msgs::TrackedObjectArray>("/filtered_radar_objects", 1);
    pub_merged_cluster_cloud_ = n.advertise<sensor_msgs::PointCloud2>("merged_cluster_cloud", 1);
    pub_labels_ = n.advertise<visualization_msgs::MarkerArray>("labels", 1);
    sub_twist_ = n.subscribe("/vehicle/twist", 1, &MultiObjectTracker::recvTwist, this);
    sub_detected_radar_objects_ = n.subscribe("/sensors/lrr/radar_objects", 1, &MultiObjectTracker::recvRadarObjects, this);
    sub_cloud_ = n.subscribe<sensor_msgs::PointCloud2>("/sensors/lidar/cepton/points", 10, &MultiObjectTracker::recvCloud, this);
    sub_yolo_object_ = n.subscribe("/sensors/camera/flir/stereo/left/yolo_objects", 1, &MultiObjectTracker::recvYoloObjects, this);
    sub_cam_info_ = n.subscribe("/sensors/camera/flir/stereo/left/camera_info", 1, &MultiObjectTracker::recvCameraInfo, this);

    update_timer_ = n.createTimer(ros::Duration(0.02), &MultiObjectTracker::updateTimerCallback, this);

    srv_.setCallback(boost::bind(&MultiObjectTracker::reconfig, this, _1, _2));
    
    looked_up_camera_transform_ = false;

  }

  void MultiObjectTracker::recvCloud(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    // Apply coordinate frame transformation
    if (!buffer_.canTransform("base_footprint", "cepton_0", ros::Time(0)))
    {
      return;
    }
    sensor_msgs::PointCloud2 transformed_msg;
    pcl_ros::transformPointCloud("base_footprint", *msg, transformed_msg, buffer_);

    // Copy into PCL cloud for processing
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Copy transformed_msg into input_cloud
    pcl::fromROSMsg<pcl::PointXYZ>(transformed_msg, *input_cloud);
    updateBoxesWithPoints(input_cloud, object_ekfs_, twist_.twist.angular.z, msg->header.stamp, output_cloud);
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg<pcl::PointXYZ>(*output_cloud, output_msg);
    output_msg.header = pcl_conversions::fromPCL(output_cloud->header);
    pub_merged_cluster_cloud_.publish(output_msg);
  }

  void MultiObjectTracker::updateTimerCallback(const ros::TimerEvent &event)
  {

    // Delete stale objects that have not been observed for a while
    std::vector<size_t> stale_objects;
    for (size_t i = 0; i < object_ekfs_.size(); i++)
    {
      object_ekfs_[i].updateFilterPredict(event.current_real, twist_.twist.angular.z);
      if (object_ekfs_[i].isStale())
      {
        stale_objects.push_back(i);
      }
    }
    for (int i = (int)stale_objects.size() - 1; i >= 0; i--)
    {
      object_ekfs_.erase(object_ekfs_.begin() + stale_objects[i]);
    }

    // Generate detected object outputs
    visualization_msgs::MarkerArray labels;
    visualization_msgs::Marker label;
    label.action = label.ADD;
    label.type = label.TEXT_VIEW_FACING;
    label.ns = "labels";
    label.scale.z = 1.0;
    label.color.a = 1.0;
    label.color.b = 1.0;
    label.color.r = 1.0;
    label.color.g = 1.0;
    label.lifetime = ros::Duration(0.1);
    avs_lecture_msgs::TrackedObjectArray object_track_msg;
    object_track_msg.header.stamp = event.current_real;
    object_track_msg.header.frame_id = "base_footprint";
    for (size_t i = 0; i < object_ekfs_.size(); i++)
    {
      if (object_ekfs_[i].getAge() < cfg_.min_age)
      {
        continue;
      }
      object_track_msg.objects.push_back(object_ekfs_[i].getEstimate());
      if (object_ekfs_[i].label_ != "")
      {
        label.header = object_track_msg.header;
        label.pose = object_track_msg.objects.back().pose;
        label.pose.position.z += 1;
        label.text = object_ekfs_[i].label_;
        label.id = object_track_msg.objects.back().id;
        labels.markers.push_back(label);
      }
    }
    pub_labels_.publish(labels);
    pub_object_tracks_.publish(object_track_msg);
  
  
  }

  void MultiObjectTracker::recvTwist(const geometry_msgs::TwistStamped& msg)
  {
    twist_ = msg;
  }

  void MultiObjectTracker::recvRadarObjects(const conti_radar_msgs::RadarObjectArrayConstPtr &msg)
  {
    if (twist_.header.stamp == ros::Time(0))
    {
      ROS_WARN("No twist received");
      return;
    }
    if (!buffer_.canTransform("base_footprint", "lrr", ros::Time(0)))
    {
      return;
    }

    // modification of recevLidarObjects to accept radar information
    //  Vector to hold the EKF indices that have already been matched to an incoming object measurement
    std::vector<int> matched_object_indices;

    // Vector to hold array indices of objects to create new EKF instances from
    std::vector<int> new_object_indices;

    // Loop through all incoming object measurements and associate them with an existing EKF, or create
    // a new EKF instance and initialize its state to the measured values
    avs_lecture_msgs::TrackedObjectArray filtered_radar;
    geometry_msgs::TransformStamped radar_to_base;
    radar_to_base = buffer_.lookupTransform("base_footprint", "lrr", ros::Time(0));
    for (size_t i = 0; i < msg->objects.size(); i++)
    {
      conti_radar_msgs::RadarObject object = msg->objects[i];
      object.position.x += object.dimensions.x / 2;
      object = applyTransformRadar(object, radar_to_base);

      if (fabs(object.position.y) > cfg_.radar_y_window)
        continue;

      tf2::Vector3 object_vel, object_pos, vehicle_linear, vehicle_angular, compensated_vel;
      tf2::fromMsg(object.velocity, object_vel);
      tf2::fromMsg(object.position, object_pos);
      tf2::fromMsg(twist_.twist.linear, vehicle_linear);
      tf2::fromMsg(twist_.twist.angular, vehicle_angular);
      compensated_vel = object_vel + vehicle_linear + vehicle_angular.cross(object_pos);

      filtered_radar.objects.emplace_back();
      filtered_radar.objects.back().header = object.header;
      filtered_radar.objects.back().bounding_box_scale = object.dimensions;
      filtered_radar.objects.back().bounding_box_scale.z = 1;
      filtered_radar.objects.back().pose.position.x = object.position.x;
      filtered_radar.objects.back().pose.position.y = object.position.y;
      filtered_radar.objects.back().pose.position.z = object.position.z;
      filtered_radar.objects.back().pose.orientation.w = 1;
      filtered_radar.objects.back().id = object.object_id;
      filtered_radar.objects.back().spawn_time = object.spawn_time;
      filtered_radar.objects.back().velocity.linear = tf2::toMsg(compensated_vel);
      
      bool bbox_match = false;
      if (looked_up_camera_transform_) {
        for (auto object: yolo_objects_.objects)
        {
          cv::Rect2d bbox = getCamBbox(filtered_radar.objects.back(), camera_transform_, model_);
          cv::Rect2d yolo_box(object.x, object.y, object.w, object.h);
          float iou = (bbox & yolo_box).area() / (bbox.area() + yolo_box.area() - (bbox & yolo_box).area());
          if (iou > 0.3)
          {
            bbox_match = true;
          }
        }
      }
      if (!bbox_match && compensated_vel.length() < cfg_.min_velocity)
      {
        continue;
      }
      // Loop through each existing EKF instance and find the one closest to the current object measurement
      double min_dist2 = INFINITY;
      int associated_track_idx = -1;
      for (size_t j = 0; j < object_ekfs_.size(); j++)
      {
        // If the current EKF instance has already been associated with a measurement, skip it and try the next one
        if (std::find(matched_object_indices.begin(), matched_object_indices.end(), j) != matched_object_indices.end())
        {
          continue;
        }

        // Compute the distance between the EKF estimate of the position and the position of the measurement
        geometry_msgs::Point est_pos = object_ekfs_[j].getEstimate().pose.position;
        tf2::Vector3 est_pos_vect(est_pos.x, est_pos.y, 0.0);
        tf2::Vector3 meas_pos_vect(object.position.x, object.position.y, 0.0);
        double d2 = (meas_pos_vect - est_pos_vect).length2();

        // If the distance is the smallest so far, mark this EKF instance as the association candidate
        if (d2 < min_dist2)
        {
          min_dist2 = d2;
          associated_track_idx = (int)j;
        }
      }

      if ((associated_track_idx < 0) || (min_dist2 > (cfg_.max_match_dist * cfg_.max_match_dist)))
      {
        // If no EKF instances exist yet, or the closest match is too far away, mark this
        // object to create a new EKF instance to track it
        new_object_indices.push_back(i);
      }
      else
      {
        // Object measurement successfully associated with an existing EKF instance...
        // Update that EKF and mark it as already associated so another object in
        //     the same measurement array doesn't also get associated to it
        object_ekfs_[associated_track_idx].radarUpdate(twist_.twist.angular.z, object);
        matched_object_indices.push_back(associated_track_idx);
      }
    }

    // After trying to associate all incoming object measurements to existing EKF instances,
    // create new EKF instances to track the inputs that weren't associated with existing ones
    for (auto new_object_idx : new_object_indices)
    {

      conti_radar_msgs::RadarObject new_radar_measurement = applyTransformRadar(msg->objects[new_object_idx], radar_to_base);

      object_ekfs_.push_back(ObjectEkf(new_radar_measurement.position.x, new_radar_measurement.velocity.x,
                                       new_radar_measurement.position.y, new_radar_measurement.velocity.y,
                                       getUniqueId(), new_radar_measurement.header.stamp, new_radar_measurement.header.frame_id));
      object_ekfs_.back().setQ(cfg_.q_pos, cfg_.q_vel);
      object_ekfs_.back().setRLidar(cfg_.lidar_r_pos);
      object_ekfs_.back().setRRadar(cfg_.radar_r_pos, cfg_.radar_r_vel);
    }
    if (filtered_radar.objects.size() > 0)
    {
      filtered_radar.header = filtered_radar.objects.back().header;
      pub_filtered_radar_.publish(filtered_radar);
    }
  }

  void MultiObjectTracker::reconfig(MultiObjectTrackerConfig &config, uint32_t level)
  {
    cfg_ = config;

    // Update Q and R matrices in each EKF instance
    for (size_t i = 0; i < object_ekfs_.size(); i++)
    {
      object_ekfs_[i].setQ(cfg_.q_pos, cfg_.q_vel);
      object_ekfs_[i].setRLidar(cfg_.lidar_r_pos);
      object_ekfs_[i].setRRadar(cfg_.radar_r_pos, cfg_.radar_r_vel);
    }
  }

  int MultiObjectTracker::getUniqueId()
  {
    int id = 0;
    bool done = false;
    while (!done)
    {
      done = true;
      for (auto &track : object_ekfs_)
      {
        if (track.getId() == id)
        {
          done = false;
          id++;
          break;
        }
      }
    }
    return id;
  }

  conti_radar_msgs::RadarObject MultiObjectTracker::applyTransformRadar(const conti_radar_msgs::RadarObject &original_object, geometry_msgs::TransformStamped transform)
  {
    conti_radar_msgs::RadarObject transformed_radar_object = original_object;
    transformed_radar_object.header.frame_id = transform.header.frame_id;
    geometry_msgs::Point pos_in, pos_out;
    pos_in.x = original_object.position.x;
    pos_in.y = original_object.position.y;
    pos_in.z = original_object.position.z;
    tf2::doTransform<geometry_msgs::Point>(pos_in, pos_out, transform);
    tf2::doTransform<geometry_msgs::Vector3>(original_object.velocity, transformed_radar_object.velocity, transform);
    tf2::doTransform<geometry_msgs::Vector3>(original_object.acceleration, transformed_radar_object.acceleration, transform);
    tf2::doTransform<geometry_msgs::Vector3>(original_object.dimensions, transformed_radar_object.dimensions, transform);
    transformed_radar_object.position.x = pos_out.x;
    transformed_radar_object.position.y = pos_out.y;
    transformed_radar_object.position.z = pos_out.z;
    return transformed_radar_object;
  
  }

void MultiObjectTracker::recvYoloObjects(const yolo_classification::YoloObjectArray& msg)
{
  // yolo_object_ = *msg;
  if (!looked_up_camera_transform_) {
    try {
      geometry_msgs::TransformStamped tf = buffer_.lookupTransform("base_footprint", msg.header.frame_id, ros::Time(0));
      tf2::fromMsg(tf.transform, camera_transform_);
      looked_up_camera_transform_ = true; // Once the lookup is successful, there is no need to keep doing the lookup
                                          // because the transform is constant
    } catch (tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(1.0, "%s", ex.what());
    }
    return;
  }
  yolo_objects_ = msg;
  for (auto object: yolo_objects_.objects)
  {
    float best_iou = 0;
    int ekf_match = -1;
    for (int i = 0; i < object_ekfs_.size(); i++)
    {
      cv::Rect2d bbox = getCamBbox(object_ekfs_[i].getEstimate(), camera_transform_, model_);
      cv::Rect2d yolo_box(object.x, object.y, object.w, object.h);
      float iou = (bbox & yolo_box).area() / (bbox.area() + yolo_box.area() - (bbox & yolo_box).area());
      if (iou > 0.3 && iou > best_iou)
      {
        best_iou = iou;
        ekf_match = i;
      }
    }
    if (ekf_match != -1) {
      object_ekfs_[ekf_match].label_ = object.label;
    }
  }
}

void MultiObjectTracker::recvCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg)
{
  camera_info_ = *msg;
  model_.fromCameraInfo(camera_info_);
}

cv::Rect2d MultiObjectTracker::getCamBbox(const avs_lecture_msgs::TrackedObject& object, const tf2::Transform& transform, const image_geometry::PinholeCameraModel& model)
{
  std::vector<double> xvals(2);
  std::vector<double> yvals(2);
  std::vector<double> zvals(2);
  xvals[0] = -0.5 * object.bounding_box_scale.x;
  xvals[1] = 0.5 * object.bounding_box_scale.x;
  yvals[0] = -0.5 * object.bounding_box_scale.y;
  yvals[1] = 0.5 * object.bounding_box_scale.y;
  zvals[0] = -0.5 * object.bounding_box_scale.z;
  zvals[1] = 0.5 * object.bounding_box_scale.z;

  int min_x = 99999;
  int max_x = 0;
  int min_y = 99999;
  int max_y = 0;
  for (size_t i = 0; i < xvals.size(); i++) {
    for (size_t j = 0; j < yvals.size(); j++) {
      for (size_t k = 0; k < zvals.size(); k++) {
        tf2::Vector3 cam_vect = transform.inverse() * tf2::Vector3(object.pose.position.x + xvals[i],
                                                                   object.pose.position.y + yvals[j],
                                                                   object.pose.position.z + zvals[k]);
        cv::Point2d p = model.project3dToPixel(cv::Point3d(cam_vect.x(), cam_vect.y(), cam_vect.z()));
        if (p.x < min_x) {
          min_x = p.x;
        }
        if (p.y < min_y) {
          min_y = p.y;
        }
        if (p.x > max_x) {
          max_x = p.x;
        }
        if (p.y > max_y) {
          max_y = p.y;
        }
      }
    }
  }

  cv::Rect2d cam_bbox(min_x, min_y, max_x - min_x, max_y - min_y);
  return cam_bbox;
}


}
