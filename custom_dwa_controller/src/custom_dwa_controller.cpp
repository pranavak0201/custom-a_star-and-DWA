#include "custom_dwa_controller/custom_dwa_controller.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"
#include "angles/angles.h"  // ✅ Added for shortest_angular_distance

namespace custom_dwa_controller
{

void CustomDWAController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  plugin_name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();

  RCLCPP_INFO(node->get_logger(), "Configuring CustomDWAController: %s", plugin_name_.c_str());

  // Get parameters
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_vel_x", rclcpp::ParameterValue(0.26));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".min_vel_x", rclcpp::ParameterValue(-0.26));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_vel_theta", rclcpp::ParameterValue(1.82));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".min_vel_theta", rclcpp::ParameterValue(-1.82));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".acc_lim_x", rclcpp::ParameterValue(2.5));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".acc_lim_theta", rclcpp::ParameterValue(3.2));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".sim_time", rclcpp::ParameterValue(1.7));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".sim_granularity", rclcpp::ParameterValue(0.025));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".angular_sim_granularity", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".vx_samples", rclcpp::ParameterValue(20.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".vth_samples", rclcpp::ParameterValue(40.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".controller_frequency", rclcpp::ParameterValue(20.0));

  node->get_parameter(plugin_name_ + ".max_vel_x", max_vel_x_);
  node->get_parameter(plugin_name_ + ".min_vel_x", min_vel_x_);
  node->get_parameter(plugin_name_ + ".max_vel_theta", max_vel_theta_);
  node->get_parameter(plugin_name_ + ".min_vel_theta", min_vel_theta_);
  node->get_parameter(plugin_name_ + ".acc_lim_x", acc_lim_x_);
  node->get_parameter(plugin_name_ + ".acc_lim_theta", acc_lim_theta_);
  node->get_parameter(plugin_name_ + ".sim_time", sim_time_);
  node->get_parameter(plugin_name_ + ".sim_granularity", sim_granularity_);
  node->get_parameter(plugin_name_ + ".angular_sim_granularity", angular_sim_granularity_);
  node->get_parameter(plugin_name_ + ".vx_samples", vx_samples_);
  node->get_parameter(plugin_name_ + ".vth_samples", vth_samples_);
  node->get_parameter(plugin_name_ + ".controller_frequency", controller_frequency_);

  // Set the specified trajectory scoring weights
  velocity_weight_ = 15.0;
  heading_weight_ = 37.0;
  obstacle_weight_ = 0.02;

  speed_limit_ = max_vel_x_;
  speed_limit_in_percentage_ = false;
}

void CustomDWAController::cleanup()
{
  auto node = node_.lock();
  RCLCPP_INFO(node->get_logger(), "Cleaning up controller: %s", plugin_name_.c_str());
}

void CustomDWAController::activate()
{
  auto node = node_.lock();
  RCLCPP_INFO(node->get_logger(), "Activating controller: %s", plugin_name_.c_str());
}

void CustomDWAController::deactivate()
{
  auto node = node_.lock();
  RCLCPP_INFO(node->get_logger(), "Deactivating controller: %s", plugin_name_.c_str());
}

void CustomDWAController::setPlan(const nav_msgs::msg::Path & path)
{
  std::lock_guard<std::mutex> lock(mutex_);
  global_plan_ = path;
}

void CustomDWAController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (percentage) {
    speed_limit_ = max_vel_x_ * speed_limit / 100.0;
  } else {
    speed_limit_ = speed_limit;
  }
  speed_limit_in_percentage_ = percentage;
}

geometry_msgs::msg::TwistStamped CustomDWAController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  auto node = node_.lock();
  std::lock_guard<std::mutex> lock(mutex_);

  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header = pose.header;

  geometry_msgs::msg::Pose2D pose_2d;
  pose_2d.x = pose.pose.position.x;
  pose_2d.y = pose.pose.position.y;
  pose_2d.theta = tf2::getYaw(pose.pose.orientation);

  auto trajectories = generateTrajectories(pose_2d, velocity);

  if (trajectories.empty()) {
    RCLCPP_WARN(node->get_logger(), "No feasible trajectories found");
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = 0.0;
    return cmd_vel;
  }

  double best_score = std::numeric_limits<double>::max();
  Trajectory best_trajectory;
  bool found_valid = false;

  for (const auto & trajectory : trajectories) {
    if (isTrajectoryCollisionFree(trajectory)) {
      double score = scoreTrajectory(trajectory, pose_2d);
      if (score < best_score) {
        best_score = score;
        best_trajectory = trajectory;
        found_valid = true;
      }
    }
  }

  if (!found_valid) {
    RCLCPP_WARN(node->get_logger(), "No collision-free trajectory found");
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = 0.0;
    return cmd_vel;
  }

  double limited_vel = std::min(best_trajectory.linear_vel, speed_limit_);
  cmd_vel.twist.linear.x = limited_vel;
  cmd_vel.twist.angular.z = best_trajectory.angular_vel;

  return cmd_vel;
}

std::vector<Trajectory> CustomDWAController::generateTrajectories(
  const geometry_msgs::msg::Pose2D & pose,
  const geometry_msgs::msg::Twist & vel)
{
  std::vector<Trajectory> trajectories;

  double min_vel_x = std::max(min_vel_x_, vel.linear.x - acc_lim_x_ / controller_frequency_);
  double max_vel_x = std::min(max_vel_x_, vel.linear.x + acc_lim_x_ / controller_frequency_);
  double min_vel_th = std::max(min_vel_theta_, vel.angular.z - acc_lim_theta_ / controller_frequency_);
  double max_vel_th = std::min(max_vel_theta_, vel.angular.z + acc_lim_theta_ / controller_frequency_);

  double vel_x_inc = (max_vel_x - min_vel_x) / vx_samples_;
  double vel_th_inc = (max_vel_th - min_vel_th) / vth_samples_;

  for (int i = 0; i < vx_samples_; ++i) {
    for (int j = 0; j < vth_samples_; ++j) {
      geometry_msgs::msg::Twist test_vel;
      test_vel.linear.x = min_vel_x + i * vel_x_inc;
      test_vel.angular.z = min_vel_th + j * vel_th_inc;

      Trajectory trajectory = simulateTrajectory(pose, test_vel, sim_time_);
      trajectories.push_back(trajectory);
    }
  }

  return trajectories;
}

Trajectory CustomDWAController::simulateTrajectory(
  const geometry_msgs::msg::Pose2D & pose,
  const geometry_msgs::msg::Twist & vel,
  double sim_time)
{
  Trajectory trajectory;
  trajectory.linear_vel = vel.linear.x;
  trajectory.angular_vel = vel.angular.z;

  geometry_msgs::msg::Pose2D current_pose = pose;
  double time = 0.0;

  while (time < sim_time) {
    trajectory.poses.push_back(current_pose);

    current_pose.x += vel.linear.x * sim_granularity_ * cos(current_pose.theta);
    current_pose.y += vel.linear.x * sim_granularity_ * sin(current_pose.theta);
    current_pose.theta += vel.angular.z * sim_granularity_;

    time += sim_granularity_;
  }

  return trajectory;
}

double CustomDWAController::scoreTrajectory(
  const Trajectory & trajectory,
  const geometry_msgs::msg::Pose2D & pose)
{
  double vel_score = velocityScore(trajectory);
  double heading_score = headingScore(trajectory, pose);
  double obstacle_score = obstacleScore(trajectory);

  return velocity_weight_ * vel_score + 
         heading_weight_ * heading_score + 
         obstacle_weight_ * obstacle_score;
}

double CustomDWAController::velocityScore(const Trajectory & trajectory)
{
  return (max_vel_x_ - std::abs(trajectory.linear_vel)) / (max_vel_x_ + 0.001);
}

double CustomDWAController::headingScore(
  const Trajectory & trajectory, 
  const geometry_msgs::msg::Pose2D & pose)
{
  if (trajectory.poses.empty()) return 1.0;

  geometry_msgs::msg::Pose2D local_goal = getLocalGoal(pose);
  
  const auto & final_pose = trajectory.poses.back();
  
  double goal_angle = atan2(local_goal.y - final_pose.y, local_goal.x - final_pose.x);
  
  // ✅ Corrected usage of shortest_angular_distance
  double heading_error = std::abs(angles::shortest_angular_distance(final_pose.theta, goal_angle));
  
  return heading_error / M_PI;
}

double CustomDWAController::obstacleScore(const Trajectory & trajectory)
{
  if (trajectory.poses.empty()) return 1.0;

  double min_distance = std::numeric_limits<double>::max();
  
  for (const auto & pose : trajectory.poses) {
    double dist = getDistanceToObstacle(pose.x, pose.y);
    min_distance = std::min(min_distance, dist);
  }

  return 1.0 / (min_distance + 0.1);
}

geometry_msgs::msg::Pose2D CustomDWAController::getLocalGoal(const geometry_msgs::msg::Pose2D & pose)
{
  geometry_msgs::msg::Pose2D local_goal = pose;

  if (global_plan_.poses.empty()) return local_goal;

  double min_dist = std::numeric_limits<double>::max();
  size_t closest_index = 0;

  for (size_t i = 0; i < global_plan_.poses.size(); ++i) {
    double dx = global_plan_.poses[i].pose.position.x - pose.x;
    double dy = global_plan_.poses[i].pose.position.y - pose.y;
    double dist = sqrt(dx * dx + dy * dy);
    
    if (dist < min_dist) {
      min_dist = dist;
      closest_index = i;
    }
  }

  size_t lookahead_index = std::min(closest_index + 10, global_plan_.poses.size() - 1);

  local_goal.x = global_plan_.poses[lookahead_index].pose.position.x;
  local_goal.y = global_plan_.poses[lookahead_index].pose.position.y;
  local_goal.theta = tf2::getYaw(global_plan_.poses[lookahead_index].pose.orientation);

  return local_goal;
}

bool CustomDWAController::isTrajectoryCollisionFree(const Trajectory & trajectory)
{
  for (const auto & pose : trajectory.poses) {
    unsigned int mx, my;
    if (!costmap_->worldToMap(pose.x, pose.y, mx, my)) {
      return false;
    }

    if (costmap_->getCost(mx, my) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
      return false;
    }
  }
  return true;
}

double CustomDWAController::getDistanceToObstacle(double x, double y)
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(x, y, mx, my)) {
    return 0.0;
  }

  double min_dist = std::numeric_limits<double>::max();
  
  int search_radius = 50;
  for (int dx = -search_radius; dx <= search_radius; ++dx) {
    for (int dy = -search_radius; dy <= search_radius; ++dy) {
      unsigned int check_x = mx + dx;
      unsigned int check_y = my + dy;
      
      if (check_x >= costmap_->getSizeInCellsX() || check_y >= costmap_->getSizeInCellsY()) {
        continue;
      }

      if (costmap_->getCost(check_x, check_y) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        double dist = sqrt(dx * dx + dy * dy) * costmap_->getResolution();
        min_dist = std::min(min_dist, dist);
      }
    }
  }

  return min_dist == std::numeric_limits<double>::max() ? 10.0 : min_dist;
}

}  // namespace custom_dwa_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(custom_dwa_controller::CustomDWAController, nav2_core::Controller)
