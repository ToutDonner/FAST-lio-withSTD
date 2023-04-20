#ifndef INTEGRATIONBASE_H_
#define INTEGRATIONBASE_H_

#include <Eigen/Eigen>
#include "math_tools.h"

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12,
};

class Preintegration
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Preintegration() = delete;

    // 预积分构造函数
    Preintegration(const Eigen::Vector3d &acc0, const Eigen::Vector3d &gyr0,
                   const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
        : acc0_{acc0},
          gyr0_{gyr0},
          linearized_acc_{acc0},
          linearized_gyr_{gyr0},
          linearized_ba_{linearized_ba},
          linearized_bg_{linearized_bg},
          jacobian_{Eigen::Matrix<double, 15, 15>::Identity()},
          sum_dt_{0.0},
          delta_p_{Eigen::Vector3d::Zero()},
          delta_q_{Eigen::Quaterniond::Identity()},
          delta_v_{Eigen::Vector3d::Zero()}
    {

        double acc_n = 0.00059;
        double gyr_n = 0.000061;
        double acc_w = 0.000011;
        double gyr_w = 0.000001;

        // nh.param<double>("/IMU/acc_n", acc_n, 0.00059);
        // nh.param<double>("/IMU/gyr_n", gyr_n, 0.000061);
        // nh.param<double>("/IMU/acc_w", acc_w, 0.000011);
        // nh.param<double>("/IMU/gyr_w", gyr_w, 0.000001);

        // 初始化协方差矩阵
        covariance_ = 0.001 * Eigen::Matrix<double, 15, 15>::Identity();
        // 重力向量
        g_vec_ = -Eigen::Vector3d(0, 0, 9.805);

        // 观测噪声矩阵
        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        noise_.block<3, 3>(0, 0) = (acc_n * acc_n) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = (gyr_n * gyr_n) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = (acc_n * acc_n) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = (gyr_n * gyr_n) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = (acc_w * acc_w) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = (gyr_w * gyr_w) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf_.push_back(dt);
        acc_buf_.push_back(acc);
        gyr_buf_.push_back(gyr);
        Propagate(dt, acc, gyr);
    }

    void Repropagate(const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
    {
        sum_dt_ = 0.0;
        acc0_ = linearized_acc_;
        gyr0_ = linearized_gyr_;
        delta_p_.setZero();
        delta_q_.setIdentity();
        delta_v_.setZero();
        linearized_ba_ = linearized_ba;
        linearized_bg_ = linearized_bg;
        jacobian_.setIdentity();
        covariance_.setZero();
        for (size_t i = 0; i < dt_buf_.size(); ++i)
        {
            Propagate(dt_buf_[i], acc_buf_[i], gyr_buf_[i]);
        }
    }

    void MidPointIntegration(double dt,
                             const Eigen::Vector3d &acc0, const Eigen::Vector3d &gyr0,
                             const Eigen::Vector3d &acc1, const Eigen::Vector3d &gyr1,
                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                             const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                             Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg,
                             bool update_jacobian)
    {
        // 利用姿态更新前的 q和acc ， 得到上一帧imu数据时刻对应的 acc_n
        Eigen::Vector3d un_acc_0 = delta_q * (acc0 - linearized_ba);
        // 均值：上一帧gyro和当前帧gyro
        Eigen::Vector3d un_gyr = 0.5 * (gyr0 + gyr1) - linearized_bg;
        // 姿态更新：得到当前帧imu数据时刻对应的姿态q
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);
        // 利用姿态更新后的q，以及当前帧imu的acc ，得到当前帧imu数据的 acc_n
        Eigen::Vector3d un_acc_1 = result_delta_q * (acc1 - linearized_ba);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 更新p，v
        result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;
        result_delta_v = delta_v + un_acc * dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        // 更新雅克比矩阵
        if (update_jacobian)
        {
            Eigen::Vector3d w_x = 0.5 * (gyr0 + gyr1) - linearized_bg;
            Eigen::Vector3d a_0_x = acc0 - linearized_ba;
            Eigen::Vector3d a_1_x = acc1 - linearized_ba;
            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x << 0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x << 0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x << 0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * dt * dt +
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt * dt;
            F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt * dt;
            F.block<3, 3>(0, 12) = -0.1667 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * -dt; // 这里应该是-0.25
            F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
            F.block<3, 3>(3, 12) = -Eigen::MatrixXd::Identity(3, 3) * dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * dt +
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
            F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * -dt;
            F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

            // NOTE: V = Fd * G_c
            // FIXME: verify if it is right, the 0.25 part
            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
            V.block<3, 3>(0, 0) = 0.5 * delta_q.toRotationMatrix() * dt * dt; // vins-mono 0.25
            V.block<3, 3>(0, 3) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * 0.5 * dt;
            V.block<3, 3>(0, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt * dt; // vins-mono 0.25
            V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt;
            V.block<3, 3>(3, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt;
            V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * dt;
            V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt * 0.5 * dt;
            V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt;
            V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3, 3) * dt;
            V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3, 3) * dt;

            jacobian_ = F * jacobian_;
            covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
        }
    }
    /**
     * @brief   IMU预积分传播方程
     * @Description  积分计算两个关键帧之间IMU测量的变化量：
     *               旋转delta_q 速度delta_v 位移delta_p
     *               加速度的biaslinearized_ba 陀螺仪的Bias linearized_bg
     *               同时维护更新预积分的Jacobian和Covariance,计算优化时必要的参数
     * @param[in]   _dt 时间间隔
     * @param[in]   _acc_1 线加速度
     * @param[in]   _gyr_1 角速度
     * @return  void
     */

    void Propagate(double dt, const Eigen::Vector3d &acc1, const Eigen::Vector3d &gyr1)
    {
        // 更新当前数据
        dt_ = dt;
        acc1_ = acc1;
        gyr1_ = gyr1;
        Eigen::Vector3d result_delta_p;
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_v;
        Eigen::Vector3d result_linearized_ba;
        Eigen::Vector3d result_linearized_bg;

        // 中值积分
        MidPointIntegration(dt,
                            acc0_, gyr0_, // 积分起点的acc，gyro
                            acc1, gyr1,
                            delta_p_, delta_q_, delta_v_, // 到积分起点的 p q v
                            linearized_ba_, linearized_bg_,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, true);

        // 更新到积分起点的 p q v
        delta_p_ = result_delta_p;
        delta_q_ = result_delta_q;
        delta_v_ = result_delta_v;
        linearized_ba_ = result_linearized_ba;
        linearized_bg_ = result_linearized_bg;
        delta_q_.normalize();
        sum_dt_ += dt_;
        acc0_ = acc1_;
        gyr0_ = gyr1_;
    }

    // 计算预积分误差
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi,
                                          const Eigen::Quaterniond &Qi,
                                          const Eigen::Vector3d &Vi,
                                          const Eigen::Vector3d &Bai,
                                          const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj,
                                          const Eigen::Quaterniond &Qj,
                                          const Eigen::Vector3d &Vj,
                                          const Eigen::Vector3d &Baj,
                                          const Eigen::Vector3d &Bgj)
    {
        // NOTE: low cost update jacobian here

        Eigen::Matrix<double, 15, 1> residuals;

        residuals.setZero();

        /// 认为更新的时候，会调整 ba和bg，因此预积分误差对ba和bg的雅克比也需要调整
        /// 另外，ba,bg的线性化点也调整
        Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba_;
        Eigen::Vector3d dbg = Bgi - linearized_bg_; // NOTE: optimized one minus the linearized one

        /// 得到更新后的预积分
        Eigen::Quaterniond corrected_delta_q = delta_q_ * deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

        /// 计算预积分误差
        residuals.block<3, 1>(O_P, 0) =
            Qi.inverse() * (-0.5 * g_vec_ * sum_dt_ * sum_dt_ + Pj - Pi - Vi * sum_dt_) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2.0 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).normalized().vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (-g_vec_ * sum_dt_ + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        return residuals;
    }

    double dt_;
    Eigen::Vector3d acc0_, gyr0_; // 初始化的acc和gyro
    Eigen::Vector3d acc1_, gyr1_;

    const Eigen::Vector3d linearized_acc_, linearized_gyr_;
    Eigen::Vector3d linearized_ba_, linearized_bg_;

    Eigen::Matrix<double, 15, 15> jacobian_, covariance_;
    Eigen::Matrix<double, 18, 18> noise_;

    double sum_dt_;
    Eigen::Vector3d delta_p_;
    Eigen::Quaterniond delta_q_;
    Eigen::Vector3d delta_v_;

    std::vector<double> dt_buf_;
    std::vector<Eigen::Vector3d> acc_buf_;
    std::vector<Eigen::Vector3d> gyr_buf_;

    Eigen::Vector3d g_vec_;
    double nf, cf;
    double acc_n;
    double gyr_n;
    double acc_w;
    double gyr_w;
    ros::NodeHandle nh;
};

#endif // INTEGRATIONBASE_H_
