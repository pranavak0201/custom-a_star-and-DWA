#include "custom_a_star_planner/custom_a_star_planner.hpp"

#include <cmath>
#include <string>
#include <memory>
#include <algorithm>

#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace custom_a_star_planner
{

void CustomAStarPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  global_frame_ = costmap_ros_->getGlobalFrameID();

  RCLCPP_INFO(
    node_->get_logger(), "CustomAStarPlanner: Name is %s", name_.c_str());
}

void CustomAStarPlanner::cleanup()
{
  RCLCPP_INFO(
    node_->get_logger(), "CleaningUp plugin %s of type CustomAStarPlanner",
    name_.c_str());
}

void CustomAStarPlanner::activate()
{
  RCLCPP_INFO(
    node_->get_logger(), "Activating plugin %s of type CustomAStarPlanner",
    name_.c_str());
}

void CustomAStarPlanner::deactivate()
{
  RCLCPP_INFO(
    node_->get_logger(), "Deactivating plugin %s of type CustomAStarPlanner",
    name_.c_str());
}

nav_msgs::msg::Path CustomAStarPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path global_path;
  
  // Check if the goal and start are in the global frame
  if (start.header.frame_id != global_frame_) {
    RCLCPP_ERROR(
      node_->get_logger(), "Planner will only except start position from %s frame",
      global_frame_.c_str());
    return global_path;
  }

  if (goal.header.frame_id != global_frame_) {
    RCLCPP_ERROR(
      node_->get_logger(), "Planner will only except goal position from %s frame",
      global_frame_.c_str());
    return global_path;
  }

  global_path.poses.clear();
  global_path.header.stamp = node_->now();
  global_path.header.frame_id = global_frame_;

  // Convert world coordinates to map coordinates
  unsigned int start_x, start_y, goal_x, goal_y;
  if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y)) {
    RCLCPP_WARN(
      node_->get_logger(), "Start position is out of map bounds");
    return global_path;
  }

  if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
    RCLCPP_WARN(
      node_->get_logger(), "Goal position is out of map bounds");
    return global_path;
  }
  
  // A* algorithm implementation
  std::priority_queue<Node*, std::vector<Node*>, NodeComparator> open_list;
  std::unordered_map<int, std::unordered_map<int, bool>> closed_list;
  std::unordered_map<int, std::unordered_map<int, Node*>> all_nodes;

  // Create start node
  Node* start_node = new Node(start_x, start_y);
  start_node->g = 0;
  start_node->h = manhattanDistance(start_x, start_y, goal_x, goal_y);
  start_node->f = start_node->g + start_node->h;

  open_list.push(start_node);
  all_nodes[start_x][start_y] = start_node;

  while (!open_list.empty()) {
    Node* current = open_list.top();
    open_list.pop();

    if (current->x == static_cast<int>(goal_x) && current->y == static_cast<int>(goal_y)) {
      // Goal reached
      global_path = reconstructPath(current);
      
      // Cleanup
      for (auto& row : all_nodes) {
        for (auto& node_pair : row.second) {
          delete node_pair.second;
        }
      }
      
      return global_path;
    }

    closed_list[current->x][current->y] = true;

    // Get five directions based on goal direction
    auto directions = getFiveDirections(current->x, current->y, goal_x, goal_y);

    for (const auto& dir : directions) {
      int new_x = current->x + dir.first;
      int new_y = current->y + dir.second;

      if (!isValid(new_x, new_y) || closed_list[new_x][new_y]) {
        continue;
      }

      double tentative_g = current->g + sqrt(dir.first * dir.first + dir.second * dir.second);

      Node* neighbor;
      if (all_nodes[new_x].find(new_y) == all_nodes[new_x].end()) {
        neighbor = new Node(new_x, new_y);
        all_nodes[new_x][new_y] = neighbor;
      } else {
        neighbor = all_nodes[new_x][new_y];
      }

      if (tentative_g < neighbor->g || neighbor->g == 0) {
        neighbor->parent = current;
        neighbor->g = tentative_g;
        neighbor->h = manhattanDistance(new_x, new_y, goal_x, goal_y);
        
        // Calculate dynamic weight P
        double P = calculateDynamicWeight(new_x, new_y, start_x, start_y, goal_x, goal_y);
        double weight = exp(P);
        
        neighbor->f = neighbor->g + weight * neighbor->h;
        
        open_list.push(neighbor);
      }
    }
  }

  // Cleanup if no path found
  for (auto& row : all_nodes) {
    for (auto& node_pair : row.second) {
      delete node_pair.second;
    }
  }

  RCLCPP_WARN(node_->get_logger(), "Failed to find a path from start to goal");
  return global_path;
}

double CustomAStarPlanner::manhattanDistance(int x1, int y1, int x2, int y2)
{
  return abs(x1 - x2) + abs(y1 - y2);
}

double CustomAStarPlanner::calculateDynamicWeight(int xn, int yn, int xs, int ys, int xg, int yg)
{
  double numerator = abs(xn - xg) + abs(yn - yg);
  double denominator = abs(xs - xg) + abs(ys - yg);
  
  if (denominator == 0) return 0.0;
  
  double P = numerator / denominator;
  return std::min(1.0, std::max(0.0, P));
}

std::vector<std::pair<int, int>> CustomAStarPlanner::getFiveDirections(int x, int y, int goal_x, int goal_y)
{
  // All 8 possible directions
  std::vector<std::pair<int, int>> all_directions = {
    {-1, -1}, {-1, 0}, {-1, 1},
    {0, -1},           {0, 1},
    {1, -1},  {1, 0},  {1, 1}
  };

  // Calculate angle to goal
  double dx = goal_x - x;
  double dy = goal_y - y;
  double angle_to_goal = atan2(dy, dx);

  // Score each direction based on alignment with goal
  std::vector<std::pair<double, std::pair<int, int>>> scored_directions;
  
  for (const auto& dir : all_directions) {
    double dir_angle = atan2(dir.second, dir.first);
    double angle_diff = abs(angle_to_goal - dir_angle);
    if (angle_diff > M_PI) angle_diff = 2 * M_PI - angle_diff;
    
    scored_directions.push_back({angle_diff, dir});
  }

  // Sort by score (smallest angle difference first)
  std::sort(scored_directions.begin(), scored_directions.end());

  // Return top 5 directions
  std::vector<std::pair<int, int>> result;
  for (size_t i = 0; i < 5 && i < scored_directions.size(); ++i) {
    result.push_back(scored_directions[i].second);
  }

  return result;
}

bool CustomAStarPlanner::isValid(int x, int y)
{
  if (x < 0 || x >= static_cast<int>(costmap_->getSizeInCellsX()) ||
      y < 0 || y >= static_cast<int>(costmap_->getSizeInCellsY())) {
    return false;
  }

  return costmap_->getCost(x, y) < nav2_costmap_2d::LETHAL_OBSTACLE;
}

nav_msgs::msg::Path CustomAStarPlanner::reconstructPath(Node* goal_node)
{
  nav_msgs::msg::Path path;
  path.header.stamp = node_->now();
  path.header.frame_id = global_frame_;

  std::vector<Node*> path_nodes;
  Node* current = goal_node;
  
  while (current != nullptr) {
    path_nodes.push_back(current);
    current = current->parent;
  }

  std::reverse(path_nodes.begin(), path_nodes.end());

  for (const auto& node : path_nodes) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;

    double world_x, world_y;
    costmap_->mapToWorld(node->x, node->y, world_x, world_y);
    
    pose.pose.position.x = world_x;
    pose.pose.position.y = world_y;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;

    path.poses.push_back(pose);
  }

  return path;
}

}  // namespace custom_a_star_planner
PLUGINLIB_EXPORT_CLASS(custom_a_star_planner::CustomAStarPlanner, nav2_core::GlobalPlanner)