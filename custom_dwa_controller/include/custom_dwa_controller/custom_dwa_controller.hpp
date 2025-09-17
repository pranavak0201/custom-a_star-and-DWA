#ifndef CUSTOM_DWA_CONTROLLER__CUSTOM_DWA_CONTROLLER_HPP_
#define CUSTOM_DWA_CONTROLLER__CUSTOM_DWA_CONTROLLER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "tf2/utils.h"

namespace custom_dwa_controller
{

struct Trajectory
{
  double linear_vel;
  double angular_vel;
  double cost;
  std::vector<geometry_msgs::msg::Pose2D> poses;
};

class CustomDWAController : public nav2_core::Controller
{
public:
  CustomDWAController() = default;
  ~CustomDWAController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

protected:
  nav2_util::LifecycleNode::WeakPtr node_; // Corrected
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::string plugin_name_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;
  
  nav_msgs::msg::Path global_plan_;
  std::mutex mutex_;
  
  // DWA parameters
  double max_vel_x_;
  double min_vel_x_;
  double max_vel_theta_;
  double min_vel_theta_;
  double acc_lim_x_;
  double acc_lim_theta_;
  double sim_time_;
  double sim_granularity_;
  double angular_sim_granularity_;
  double vx_samples_;
  double vth_samples_;
  double controller_frequency_;
  
  // Trajectory scoring weights (as specified)
  double velocity_weight_;     // α = 15
  double heading_weight_;      // β = 37  
  double obstacle_weight_;     // γ = 0.02
  
  double speed_limit_;
  bool speed_limit_in_percentage_;

  std::vector<Trajectory> generateTrajectories(
    const geometry_msgs::msg::Pose2D & pose,
    const geometry_msgs::msg::Twist & vel);

  double scoreTrajectory(
    const Trajectory & trajectory,
    const geometry_msgs::msg::Pose2D & pose);

  double velocityScore(const Trajectory & trajectory);
  double headingScore(const Trajectory & trajectory, const geometry_msgs::msg::Pose2D & pose);
  double obstacleScore(const Trajectory & trajectory);

  geometry_msgs::msg::Pose2D getLocalGoal(const geometry_msgs::msg::Pose2D & pose);
  
  Trajectory simulateTrajectory(
    const geometry_msgs::msg::Pose2D & pose,
    const geometry_msgs::msg::Twist & vel,
    double sim_time);

  bool isTrajectoryCollisionFree(const Trajectory & trajectory);
  
  double getDistanceToObstacle(double x, double y);
};

}  // namespace custom_dwa_controller

#endif