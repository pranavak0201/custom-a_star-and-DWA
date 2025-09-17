#ifndef CUSTOM_A_STAR_PLANNER__CUSTOM_A_STAR_PLANNER_HPP_
#define CUSTOM_A_STAR_PLANNER__CUSTOM_A_STAR_PLANNER_HPP_

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_util/lifecycle_node.hpp"

namespace custom_a_star_planner
{

struct Node
{
  int x, y;
  double g, h, f;
  Node* parent;
  
  Node(int x_, int y_) : x(x_), y(y_), g(0), h(0), f(0), parent(nullptr) {}
};

struct NodeComparator
{
  bool operator()(const Node* a, const Node* b) const
  {
    return a->f > b->f;
  }
};

class CustomAStarPlanner : public nav2_core::GlobalPlanner
{
public:
  CustomAStarPlanner() = default;
  ~CustomAStarPlanner() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;

  void activate() override;

  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  std::shared_ptr<tf2_ros::Buffer> tf_;
  nav2_util::LifecycleNode::SharedPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;
  std::string global_frame_, name_;

  double manhattanDistance(int x1, int y1, int x2, int y2);
  double calculateDynamicWeight(int xn, int yn, int xs, int ys, int xg, int yg);
  std::vector<std::pair<int, int>> getFiveDirections(int x, int y, int goal_x, int goal_y);
  bool isValid(int x, int y);
  nav_msgs::msg::Path reconstructPath(Node* goal_node);
};

}  // namespace custom_a_star_planner

#endif