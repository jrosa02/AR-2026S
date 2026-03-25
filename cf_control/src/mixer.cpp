// Copyright 2026 Hubert Szolc, EVS Group, AGH

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "cf_control/mixer.hpp"

#include <cmath>

namespace evs
{
namespace cf
{
Mixer::Mixer(const rclcpp::NodeOptions & options)
: Node("mixer", options),
  desired_thrust_(0.0),
  desired_roll_torque_(0.0),
  desired_pitch_torque_(0.0),
  desired_yaw_torque_(0.0)
{
  // Parameters
  this->declare_parameter("process_rate", 200.0);              // Hz
  this->declare_parameter("arm_length", 0.046);                // meters
  this->declare_parameter("thrust_coefficient", 1.28192e-08);  // N/((rad/s)^2)
  this->declare_parameter("drag_coefficient", 8.06428e-05);    // Nm/((rad/s)^2)

  double process_rate = this->get_parameter("process_rate").as_double();
  double arm_length = this->get_parameter("arm_length").as_double();
  double thrust_coefficient = this->get_parameter("thrust_coefficient").as_double();
  double drag_coefficient = this->get_parameter("drag_coefficient").as_double();
  double torque_arm = arm_length / std::sqrt(2);  // Torque arm for X configuration

  // Compute coefficients for the mixer
  k_thrust_ = 1 / thrust_coefficient;
  k_torque_ = 1 / (thrust_coefficient * torque_arm);
  k_drag_ = 1 / drag_coefficient;

  // ROS interfaces
  motor_command_publisher_ =
    this->create_publisher<actuator_msgs::msg::Actuators>("/crazyflie/motor_speed", 10);
  control_command_subscriber_ = this->create_subscription<cf_control_msgs::msg::ThrustAndTorque>(
    "/cf_control/control_command", 10,
    std::bind(&Mixer::control_command_callback, this, std::placeholders::_1));

  // Kick off the processing function at the specified rate
  process_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / process_rate), std::bind(&Mixer::process, this));

  RCLCPP_INFO(this->get_logger(), "Mixer node has been started.");
}

void Mixer::control_command_callback(const cf_control_msgs::msg::ThrustAndTorque::SharedPtr msg)
{
  std::scoped_lock lock(input_mutex_);
  desired_thrust_ = msg->collective_thrust;
  desired_roll_torque_ = msg->torque.x;
  desired_pitch_torque_ = msg->torque.y;
  desired_yaw_torque_ = msg->torque.z;
}

// TODO: Add safety check for input stream to ensure that we don't process stale commands
// (e.g., if the subscriber callback hasn't been called for a while, we might want to set desired
// commands to zero or some safe default)
void Mixer::process()
{
  // Read the current input commands (thread-safe)
  double thrust, roll_torque, pitch_torque, yaw_torque;
  {
    std::scoped_lock lock(input_mutex_);
    thrust = desired_thrust_;
    roll_torque = desired_roll_torque_;
    pitch_torque = desired_pitch_torque_;
    yaw_torque = desired_yaw_torque_;
  }

  // Compute the desired motor speeds based on the mixer logic
  double thrust_part = k_thrust_ * thrust;
  double roll_part = k_torque_ * roll_torque;
  double pitch_part = k_torque_ * pitch_torque;
  double yaw_part = k_drag_ * yaw_torque;

  double motor_speed_1 =
    0.5 * std::sqrt(thrust_part - roll_part - pitch_part - yaw_part);  // front right
  double motor_speed_2 =
    0.5 * std::sqrt(thrust_part - roll_part + pitch_part + yaw_part);  // rear right
  double motor_speed_3 =
    0.5 * std::sqrt(thrust_part + roll_part + pitch_part - yaw_part);  // rear left
  double motor_speed_4 =
    0.5 * std::sqrt(thrust_part + roll_part - pitch_part + yaw_part);  // front left

  // Ensure motor speeds are non-negative (since we can't have negative speeds)
  motor_speed_1 = std::max(0.0, motor_speed_1);
  motor_speed_2 = std::max(0.0, motor_speed_2);
  motor_speed_3 = std::max(0.0, motor_speed_3);
  motor_speed_4 = std::max(0.0, motor_speed_4);

  // Publish the computed motor speeds
  actuator_msgs::msg::Actuators motor_command;
  motor_command.header.stamp = this->now();
  motor_command.velocity = {motor_speed_1, motor_speed_2, motor_speed_3, motor_speed_4};
  motor_command_publisher_->publish(motor_command);

  RCLCPP_INFO(
    this->get_logger(), "Published motor speeds: [%.2f, %.2f, %.2f, %.2f]", motor_speed_1,
    motor_speed_2, motor_speed_3, motor_speed_4);
}
}  // namespace cf
}  // namespace evs

RCLCPP_COMPONENTS_REGISTER_NODE(evs::cf::Mixer)
