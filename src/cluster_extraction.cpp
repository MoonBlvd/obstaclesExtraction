#include <stdlib.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/lexical_cast.hpp>

using namespace cv;

struct minmax_t
{
  float x, y, z;
};
int main (int argc, char** argv)
{
  // Read in the full image
  
  Mat img = imread("../data/image0771.ppm", 1); //read the image data in the file "MyPic.JPG" and store it in 'img'
  Mat imgResized;
  Mat imgRotated;
  resize(img, imgResized, Size(), 0.4, 0.2, INTER_LINEAR);
  
  Point2f pc(imgResized.cols/2.0, imgResized.rows/2.0);
  Mat M = getRotationMatrix2D(pc, -90, 1.0);
  Size s = imgResized.size();
  warpAffine(imgResized, imgRotated, M, Size(s.height,s.width));
  
  resize(imgRotated, imgResized, Size(), 2.1, 0.5, INTER_LINEAR);
  namedWindow("fullImage", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
  imshow("fullImage", imgResized);
  waitKey(0);

  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  //reader.read ("../data/table_scene_lms400.pcd", *cloud);
  reader.read ("../data/Scan1000Cam4.pcd", *cloud);
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.01f, 0.01f, 0.01f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  
  pcl::visualization::PCLVisualizer viewer;

  std::cout << "nr_points:d"<< nr_points << std::endl;

  std::cout << "Size of the filtered points: "  << cloud_filtered->points.size() << std::endl;

  time_t start = time(0);
  while (cloud_filtered->points.size () > 0.2 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }
  double timeDiff = difftime( time(0), start);
  std::cout << "CPU Time of gound points classification: "  << timeDiff << std::endl;

  std::cout << "Size of the above ground points: "  << cloud_filtered->points.size() << std::endl;
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.5); // 50cm
  ec.setMinClusterSize (20);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  pcl::PointCloud<pcl::PointXYZ>::Ptr all_Obs (new pcl::PointCloud<pcl::PointXYZ>);

  viewer.addPointCloud(cloud, "Original");
  viewer.spin();
  viewer.removePointCloud("Original");
  viewer.addPointCloud(cloud_filtered, "aboveGround");
  viewer.spin();
  viewer.removePointCloud("aboveGround");
  int j = 0;
  // std::cout << "The begin of cluster indices:" << cluster_indices.begin () << std::endl;
  // std::cout << "The end of cluster indices:" << cluster_indices.end () << std::endl;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    // Save point cloud to .pcd
    // std::stringstream ss;
    // ss << "cloud_cluster_" << j << ".pcd";
    // writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
    j++;
    // if(j-1!=0)
    //   viewer.removePointCloud(boost::lexical_cast<std::string>(j-1));


    *all_Obs += *cloud_cluster;
    // Draw bounding box;
    minmax_t min;
    minmax_t max;
    min.x = min.y = min.z = 1000;
    max.x = max.y = max.z = -1000;
    for (int i = 0; i < cloud_cluster->width; i++){
      //std::cout<< "x = " << cloud_cluster->points[1].x << std::endl;
      if (cloud_cluster->points[i].x < min.x){
        min.x = cloud_cluster->points[i].x;
      }
      if (cloud_cluster->points[i].y < min.y){
        min.y = cloud_cluster->points[i].y;
      }
      if (cloud_cluster->points[i].z < min.z){
        min.z = cloud_cluster->points[i].z;
      }
      if (cloud_cluster->points[i].x > max.x){
        max.x = cloud_cluster->points[i].x;
      }
      if (cloud_cluster->points[i].y > max.y){
        max.y = cloud_cluster->points[i].y;
      }
      if (cloud_cluster->points[i].z > max.z){
        max.z = cloud_cluster->points[i].z;
      }
    }
    viewer.addPointCloud(cloud_cluster, boost::lexical_cast<std::string>(j));
    // pcl::getMinMax3D(cloud_cluster, min_pt, max_pt);
    viewer.addCube(min.x, max.x, min.y, max.y, min.z, max.z, 1.0, 0.0, 0.0, boost::lexical_cast<std::string>(j));
    viewer.spin();
    // cloud_cluster->drawTBoundingBox(viewer, j);
    if (j == 1){
      std::cout << "We entered the obstacle extraction loop." << std::endl;
    }
  }
  // Save point cloud to .pcd
    std::stringstream ss;
    ss << "clusterCam4Size20.pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *all_Obs, false); //*

  if (j == 0){
    std::cout << "We skipped the obstacle extraction loop." << std::endl;
  }
  else{
    std::cout << "We finished the obstacle extraction loop." << std::endl;

  }

  return (0);
}