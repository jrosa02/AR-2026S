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

#ifndef CF_CONTROL_MIXER_HPP_
#define CF_CONTROL_MIXER_HPP_

#include <actuator_msgs/msg/actuators.hpp>
#include <cf_control_msgs/msg/thrust_and_torque.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace evs
{
namespace cf
{
class Mixer : public rclcpp::Node
{
public:
  explicit Mixer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  /**
  * @brief Main processing function for the mixer.
  *
  * This function is called at a fixed rate (defined by the "process_rate" parameter) and is responsible for:
  * 1. Reading the current input control commands (torque and thrust) from the internal state.
  * 2. Applying the mixer logic to compute the desired motor speeds based on the input commands and the mixer coefficients.
  * 3. Publishing the computed motor speeds to the appropriate topic for the motor controller to consume.
  *
  */
  void process();
  rclcpp::TimerBase::SharedPtr process_timer_;

  // ROS interfaces
  rclcpp::Publisher<actuator_msgs::msg::Actuators>::SharedPtr motor_command_publisher_;
  rclcpp::Subscription<cf_control_msgs::msg::ThrustAndTorque>::SharedPtr
    control_command_subscriber_;

  // Callbacks
  /**
   * @brief Callback function for receiving control commands (thrust and torque).
   *
   * @param msg The incoming message containing the desired thrust and torque values.
   */
  void control_command_callback(const cf_control_msgs::msg::ThrustAndTorque::SharedPtr msg);

  // Mixer parameters
  double k_thrust_;
  double k_torque_;
  double k_drag_;

  // Input control commands (to be updated by the subscriber callback)
  double desired_thrust_;
  double desired_roll_torque_;
  double desired_pitch_torque_;
  double desired_yaw_torque_;
  std::mutex input_mutex_;
};
}  // namespace cf
}  // namespace evs

#endif  // CF_CONTROL_MIXER_HPP_
