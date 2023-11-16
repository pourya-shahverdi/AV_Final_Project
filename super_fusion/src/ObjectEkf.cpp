#include "ObjectEkf.hpp"

namespace super_fusion
{

  ObjectEkf::ObjectEkf(double x_pos0, double x_vel0,
                       double y_pos0, double y_vel0, int id,
                       const ros::Time &t0, const std::string &frame_id)
  {

    // Initialize estimate covariance to identity
    P_.setIdentity();

    // Initialize state estimate to input arguments
    X_ << x_pos0, x_vel0, y_pos0, y_vel0;

    // Initialize all time stamps to the starting time
    estimate_stamp_ = t0;
    measurement_stamp_ = t0;
    spawn_stamp_ = t0;

    // Set dummy values for Q and R. These should be set from the code
    // instantiating the ObjectEkf class using setQ() and setR()
    setQ(1.0, 1.0);
    setRLidar(1.0);
    setRRadar(1.0, 2.5);

    frame_id_ = frame_id;
    id_ = id;
  }

  void ObjectEkf::updateFilterPredict(const ros::Time &current_time, double yaw_rate)
  {
    // Calculate time difference between current time and filter state
    double dt = (current_time - estimate_stamp_).toSec();
    if (fabs(dt) > 2)
    {
      // Large time jump detected... just reset to the current time
      spawn_stamp_ = current_time;
      estimate_stamp_ = current_time;
      measurement_stamp_ = current_time;
      return;
    }

    // Propagate estimate prediction and update estimate with result
    StateMatrix A = stateJacobian(dt, yaw_rate, X_);
    X_ = statePrediction(dt, yaw_rate, X_);
    P_ = covPrediction(A, Q_, P_);
    estimate_stamp_ = current_time;
  }

  StateVector ObjectEkf::peekState(ros::Time time, double yaw_rate)
  {
    return statePrediction((time - estimate_stamp_).toSec(), yaw_rate, X_);
  }

  void ObjectEkf::radarUpdate(double yaw_rate, const conti_radar_msgs::RadarObject &meas)
  {
    // Calculate time difference between measurement and filter state
    double dt = (meas.header.stamp - estimate_stamp_).toSec();
    if (fabs(dt) > 2)
    {
      // Large time jump detected... reset filter to this measurement
      X_ << meas.position.x, 0.0, meas.position.y, 0.0;
      P_.setIdentity();
      spawn_stamp_ = meas.header.stamp;
      estimate_stamp_ = meas.header.stamp;
      measurement_stamp_ = meas.header.stamp;
      return;
    }

    // Prediction step
    StateMatrix A = stateJacobian(dt, yaw_rate, X_);
    StateVector predicted_state = statePrediction(dt, yaw_rate, X_);
    StateMatrix predicted_cov = covPrediction(A, Q_, P_);

    // Measurement update
    // Define measurement matrix
    Eigen::Matrix4d C;
    C.row(0) << 1, 0, 0, 0;
    C.row(1) << 0, 1, 0, 0;
    C.row(2) << 0, 0, 1, 0;
    C.row(3) << 0, 0, 0, 1;

    Eigen::Vector4d meas_vect;
    meas_vect << meas.position.x, meas.velocity.x, meas.position.y, meas.velocity.y;

    // Compute expected measurement based on predicted_state
    Eigen::Vector4d expected_meas;
    expected_meas << predicted_state(0), predicted_state(1), predicted_state(2), predicted_state(3);

    // Compute residual covariance matrix
    Eigen::Matrix4d S;
    S = C * predicted_cov * C.transpose() + R_radar;

    // Compute Kalman gain
    Eigen::Matrix4d K;
    K = predicted_cov * C.transpose() * S.inverse();

    // Update state estimate
    Eigen::Vector4d measurement_residual;
    measurement_residual = meas_vect - expected_meas;
    X_ = predicted_state + K * measurement_residual;

    // Update estimate covariance
    StateMatrix I;
    I.setIdentity();
    P_ = (I - K * C) * predicted_cov;

    if (dt < 0) {
      A = stateJacobian(-dt, yaw_rate, X_);
      X_ = statePrediction(-dt, yaw_rate, X_);
      P_ = covPrediction(A, Q_, P_);
    }
    // Set estimate time stamp and latest measurement time stamp to the stamp in the input argument
    if (dt > 0)
    {
      estimate_stamp_ = meas.header.stamp;
    }
    if (meas.header.stamp > measurement_stamp_)
    {
      measurement_stamp_ = meas.header.stamp;
    }
    // copy the size and position
    // the radar will often return a z dimension of 0. for now lets side step that and just constrain everything to at least one meter
    geometry_msgs::Vector3 s;
    s.x = (meas.dimensions.x > 1.0) ? meas.dimensions.x : 4.5;
    s.y = (meas.dimensions.y > 1.0) ? meas.dimensions.y : 2.0;
    s.z = (meas.dimensions.z > 1.0) ? meas.dimensions.z : 1.0;

    // smoothly change scale between samples.
    // Entirely a visual change, but it's nice :)
    scale_.x = scale_.x * 0.9 + s.x * 0.1;
    scale_.y = scale_.y * 0.9 + s.y * 0.1;
    scale_.z = scale_.z * 0.9 + s.z * 0.1;

    z_ = 0.9 * z_ + 0.1 * meas.position.z;
    heading_ = meas.orientation_angle;
  }

  void ObjectEkf::lidarUpdate(double yaw_rate, ros::Time time, const Eigen::Vector4d& pos, int num_points)
  {
    // Calculate time difference between measurement and filter state
    double dt = (time.toSec() - estimate_stamp_.toSec());
    if (fabs(dt) > 2)
    {
      // Large time jump detected... reset filter to this measurement
      ROS_INFO("Large time jump of %f", (float)dt);
      X_ << pos(0), 0.0, pos(1), 0.0;
      P_.setIdentity();
      spawn_stamp_ = ros::Time(time);
      estimate_stamp_ = spawn_stamp_;
      measurement_stamp_ = spawn_stamp_;
      return;
    }

    // Prediction step
    StateMatrix A = stateJacobian(dt, yaw_rate, X_);
    StateVector predicted_state = statePrediction(dt, yaw_rate, X_);
    StateMatrix predicted_cov = covPrediction(A, Q_, P_);

    // Measurement update
    // Define measurement matrix
    Eigen::Matrix<double, 2, 4> C;
    C.row(0) << 1, 0, 0, 0;
    C.row(1) << 0, 0, 1, 0;

    Eigen::Vector2d meas_vect;
    meas_vect << pos(0), pos(1);

    // Compute expected measurement based on predicted_state
    Eigen::Vector2d expected_meas;
    expected_meas << predicted_state(0), predicted_state(2);

    // Compute residual covariance matrix
    Eigen::Matrix2d S;
    S = C * predicted_cov * C.transpose() + (R_lidar / ceil((float)num_points / 100.0));

    // Compute Kalman gain
    Eigen::Matrix<double, 4, 2> K;
    K = predicted_cov * C.transpose() * S.inverse();

    // Update state estimate
    Eigen::Vector2d measurement_residual;
    measurement_residual = meas_vect - expected_meas;
    X_ = predicted_state + K * measurement_residual;

    // Update estimate covariance
    StateMatrix I;
    I.setIdentity();
    P_ = (I - K * C) * predicted_cov;

    if (dt < 0) {
      A = stateJacobian(-dt, yaw_rate, X_);
      X_ = statePrediction(-dt, yaw_rate, X_);
      P_ = covPrediction(A, Q_, P_);
    }
    // Set estimate time stamp and latest measurement time stamp to the stamp in the input argument
    if (dt > 0)
    {
      estimate_stamp_ = time;
    }

    // lidar sizes are awful, lets just use the radar information.
    z_ = pos(2);
  }

  StateVector ObjectEkf::statePrediction(double dt, double yaw_rate, const StateVector &old_state)
  {
    // Propagate the old_state argument through the discrete state equations and put the results in new_state
    //        The 'dt' argument of this method is the sample time to use
    //  [x \dot{x} y \dot{y}].T
    double x_km1 = old_state(0);
    double x_dot_km1 = old_state(1);
    double y_km1 = old_state(2);
    double y_dot_km1 = old_state(3);

    StateVector new_state;

    new_state.row(0) << x_km1 + x_dot_km1 * dt + dt * (-yaw_rate * x_km1 * sin(yaw_rate * dt) + yaw_rate * y_km1 * cos(yaw_rate * dt));
    new_state.row(1) << x_dot_km1;
    new_state.row(2) << y_km1 + y_dot_km1 * dt + dt * (-yaw_rate * x_km1 * cos(yaw_rate * dt) - yaw_rate * y_km1 * sin(yaw_rate * dt));
    new_state.row(3) << y_dot_km1;

    return new_state;
  }

  StateMatrix ObjectEkf::stateJacobian(double dt, double yaw_rate, const StateVector &state)
  {
    //  Fill in the elements of the state Jacobian
    //       The 'dt' argument of this method is the sample time to use
    StateMatrix A;
    A.row(0) << -yaw_rate * dt * sin(yaw_rate * dt) + 1, dt, yaw_rate * dt * cos(yaw_rate * dt), 0;
    A.row(1) << 0, 1, 0, 0;
    A.row(2) << -yaw_rate * dt * cos(yaw_rate * dt), 0, -yaw_rate * dt * sin(yaw_rate * dt) + 1, dt;
    A.row(3) << 0, 0, 0, 1;
    return A;
  }

  StateMatrix ObjectEkf::covPrediction(const StateMatrix &A, const StateMatrix &Q, const StateMatrix &old_cov)
  {
    // Propagate the old_cov argument through the covariance prediction equation and put the result in new_cov
    StateMatrix new_cov;
    // new_cov.setZero();

    //\boldsymbol{P}_{k|k-1} = \boldsymbol{A_{k-1}}\boldsymbol{P}_{k-1|k-1}\boldsymbol{A}^T_{k-1} + \boldsymbol{Q}
    new_cov = A * old_cov * A.transpose() + Q;

    return new_cov;
  }

  // Has it been a long time since the last time the filter has been
  // updated with a measurement sample?
  bool ObjectEkf::isStale()
  {
    return (estimate_stamp_ - measurement_stamp_) > ros::Duration(0.5);
  }

  // Look up amount of time since filter was created
  double ObjectEkf::getAge()
  {
    return (estimate_stamp_ - spawn_stamp_).toSec();
  }

  // Sets the process noise standard deviations
  void ObjectEkf::setQ(double q_pos, double q_vel)
  {
    // Populate Q_ with q_pos and q_vel
    //  Q_.setZero();
    Q_.row(0) << pow(q_pos, 2), 0, 0, 0;
    Q_.row(1) << 0, pow(q_vel, 2), 0, 0;
    Q_.row(2) << 0, 0, pow(q_pos, 2), 0;
    Q_.row(3) << 0, 0, 0, pow(q_vel, 2);
  }

  // Sets the measurement noise standard deviation
  void ObjectEkf::setRLidar(double r_pos)
  {
    // populate the lidar measurement noise matrix
    R_lidar.row(0) << pow(r_pos, 2), 0;
    R_lidar.row(1) << 0, pow(r_pos, 2);
  }

  void ObjectEkf::setRRadar(double r_pos, double r_vel)
  {
    // populate the lidar measurement noise matrix
    R_radar.row(0) << pow(r_pos, 2), 0, 0, 0;
    R_radar.row(1) << 0, pow(r_vel, 2), 0, 0;
    R_radar.row(2) << 0, 0, pow(r_pos, 2), 0;
    R_radar.row(3) << 0, 0, 0, pow(r_vel, 2);
  }

  // Return the ID number property
  int ObjectEkf::getId()
  {
    return id_;
  }

  // Create and return a DetectedObject output from filter state
  avs_lecture_msgs::TrackedObject ObjectEkf::getEstimate()
  {
    avs_lecture_msgs::TrackedObject estimate_output;
    estimate_output.header.stamp = estimate_stamp_;
    estimate_output.spawn_time = spawn_stamp_;
    estimate_output.header.frame_id = frame_id_;
    estimate_output.id = id_;
    estimate_output.pose.position.z = z_;
    estimate_output.pose.orientation.w = 1.0;
    estimate_output.bounding_box_scale = scale_;

    // Populate output x and y position with filter estimate
    estimate_output.pose.position.x = X_(0);
    estimate_output.pose.position.y = X_(2);

    // Populate output x and y velocity with filter estimate
    estimate_output.velocity.linear.x = X_(1);
    estimate_output.velocity.linear.y = X_(3);

    return estimate_output;
  }

}