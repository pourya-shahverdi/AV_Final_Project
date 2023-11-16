#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include "ObjectEkf.hpp"

inline void updateBoxesWithPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                                  std::vector<super_fusion::ObjectEkf>& tracks,
                                  double yaw_rate,
                                  ros::Time stamp,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out)
{
  pcl::CropBox<pcl::PointXYZ> filter;
  pcl::IndicesPtr object_points(new pcl::Indices);
  pcl::Indices tmp_indices;
  filter.setInputCloud(cloud_in);
  for (auto& track : tracks) 
  {
    auto state = track.peekState(stamp, yaw_rate);
    filter.setTranslation(Eigen::Vector3f(state(0), state(2), track.z_));
    filter.setRotation(Eigen::Vector3f(0, 0, track.heading_));
    float velocity_factor = (1.0 + 0.05 * abs(state(1)));
    filter.setMin(Eigen::Vector4f(-0.7 * velocity_factor * track.scale_.x, -0.55 * track.scale_.y, -4.0, 1.0));
    filter.setMax(Eigen::Vector4f(0.7 * velocity_factor * track.scale_.x, 0.55 * track.scale_.y, 4.0, 1.0));
    filter.filter(tmp_indices);
    if (tmp_indices.size() > 100) 
    {
      std::priority_queue<float> min_points;
      int queue_size = tmp_indices.size() / 4;
      for (auto index : tmp_indices) 
      {
        float z = cloud_in->points[index].z;
        if (min_points.size() < queue_size) 
        {
          min_points.push(z);
        } else if (cloud_in->points[index].z < min_points.top())
        {
          min_points.push(z);
          min_points.pop();
        }
      }
      float min_z = min_points.top();
      tmp_indices.erase(std::remove_if(tmp_indices.begin(), tmp_indices.end(), [&cloud_in, min_z](int i){
        return cloud_in->points[i].z < min_z + 0.25;
      }), tmp_indices.end());
    }
    if (tmp_indices.size() > 50)
    {
      Eigen::Vector4d centroid;
      pcl::compute3DCentroid<pcl::PointXYZ>(*cloud_in, tmp_indices, centroid);
      track.lidarUpdate(yaw_rate, stamp, centroid, tmp_indices.size());
    }
    object_points->insert(object_points->end(), tmp_indices.begin(), tmp_indices.end());
  }
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud_in);
  extract.setIndices(object_points);
  extract.filter(*cloud_out);
}