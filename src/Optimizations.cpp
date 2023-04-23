#include <ros/ros.h>
#include <ros/time.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <cstdio>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include <deque>
#include <string>
#include <condition_variable>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>

#include "common_lib.h"
#include "STDesc.h"
#include "ceres/pose_graph_3d_error_term.h"
#include "ceres/loop_clousure_factor.h"

// #include "Preintegration.h"

std::mutex mtx_buffer;
std::condition_variable sig_buffer;

int sub_frame_num_;

bool flg_exit;
bool new_pcl = false, new_pose = false;
bool first_pose = true;

double last_timestamp_imu = 0,
       last_timestamp_odom = 0, last_timestamp_pcl = 0;
double odom_rcv_time = 0, pcl_rcv_time = 0;
double lio_weight, std_weight;

std::string lidar_topic, odom_topic, oriTraj_path, optTraj_path;

std::deque<sensor_msgs::ImuPtr> imu_buffer;
std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcl_buffer;
std::deque<double> time_buffer;
std::deque<nav_msgs::OdometryPtr> odom_buffer;
std::deque<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> pose_deq;
std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>,
          pcl::PointCloud<pcl::PointXYZI>::Ptr>
    data_pkg;
std::vector<std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>,
                      double>>
    pose_vec;
std::vector<std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>,
                      double>>
    pose_vec_wo_opt;

std::vector<pcl::PointCloud<pcl::PointXYZI>> pcl_wait_pub;
std::vector<int> pose_constant;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::ImuConstPtr &imuMsg)
{
    sensor_msgs::ImuPtr msg(new sensor_msgs::Imu(*imuMsg));

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void odom_cbk(const nav_msgs::OdometryConstPtr &odomMsg)
{
    nav_msgs::OdometryPtr msg(new nav_msgs::Odometry(*odomMsg));

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();
    if (timestamp < last_timestamp_odom)
    {
        ROS_WARN("odom loop back, clear buffer");
        odom_buffer.clear();
    }
    last_timestamp_odom = timestamp;

    Eigen::Vector3d p = Eigen::Vector3d(msg->pose.pose.position.x,
                                        msg->pose.pose.position.y,
                                        msg->pose.pose.position.z);
    Eigen::Quaterniond r = Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                              msg->pose.pose.orientation.x,
                                              msg->pose.pose.orientation.y,
                                              msg->pose.pose.orientation.z);
    r.normalize();
    std::pair<Eigen::Vector3d, Eigen::Quaterniond>
        pose(p, r);
    pose_deq.push_back(pose);
    new_pose = true;
    odom_rcv_time = timestamp;

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void pcl_cbk(const sensor_msgs::PointCloud2ConstPtr &pclMsg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*pclMsg, *pcl_ptr);

    double timestamp = pclMsg->header.stamp.toSec();
    if (timestamp < last_timestamp_pcl)
    {
        ROS_WARN("odom loop back, clear buffer");
        pcl_buffer.clear();
    }
    last_timestamp_pcl = timestamp;
    // cout << "[receive pcl time]" << timestamp;

    mtx_buffer.lock();
    pcl_buffer.push_back(pcl_ptr);
    time_buffer.push_back(timestamp);
    new_pcl = true;
    pcl_rcv_time = timestamp;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool package_date()
{
    if (new_pcl && new_pose)
    {
        new_pose = false;
        new_pcl = false;

        if (pcl_rcv_time == odom_rcv_time)
        {
            if ((pose_deq.size() || pcl_buffer.size()) == 0)
            {
                return false;
            }
            data_pkg = std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr>(pose_deq.front(), pcl_buffer.front());
            pose_deq.pop_front();
            pcl_buffer.pop_front();
            return true;
        }
        else
        {
            ROS_ERROR("different btwn pcl time and odom time,please check time stamp");
            printf("pcl_rcv_time:%20.2f", pcl_rcv_time);
            printf("odom_rcv_time:%20.2f", odom_rcv_time);
            return false;
        }
    }
    else
        return false;
}

/**
 * @brief ceres
 *
 */
void BuildOptimizationProblem(ceres::Problem *problem, std::vector<std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>, double>> &pose_vec, const int curFrameInd, const int assFrameInd, const int sub_frame_num_)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
    // ceres::LocalParameterization *local_parameter;
    // cout << "[Ceres info]:"
    //      << "current frame index:" << curFrameInd << ","
    //      << "associated frame index" << assFrameInd << endl;
    int j = 0;
    // for (int i = assFrameInd + 1; i < curFrameInd; i++)
    // for (int i = 0; i < curFrameInd; i++)
    for (int i = 0; i < pose_vec.size() - 1; i++)
    {

        ceres::Pose3d curPose, assPose;
        assPose.p = pose_vec[i].first.first;
        assPose.q = pose_vec[i].first.second;
        assPose.q = assPose.q.normalized();

        curPose.p = pose_vec[i + 1].first.first;
        curPose.q = pose_vec[i + 1].first.second;
        curPose.q = curPose.q.normalized();

        // cout << "[Ceres info]:" << assFrameInd << " " << assPose.p.transpose() << "," << assPose.q.coeffs().transpose() << endl;
        /**
         * @brief current frame 和 associated frame的相对位置变换
         *
         */
        ceres::Pose3d pose_c2a;
        pose_c2a.p = assPose.q.conjugate() * (curPose.p - assPose.p);
        pose_c2a.q = assPose.q.conjugate() * curPose.q;
        pose_c2a.q.normalize();

        const Eigen::Matrix<double, 6, 6> sqrt_information = (Eigen::Matrix<double, 6, 6>::Identity() * lio_weight).llt().matrixL();
        // const Eigen::Matrix<double, 6, 6> sqrt_information = (Eigen::Matrix<double, 6, 6>::Identity() * lio_weight);

        ceres::CostFunction *cost_function =
            ceres::PoseGraph3dErrorTerm::Create(pose_c2a, sqrt_information);

        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  pose_vec[i].first.first.data(),
                                  pose_vec[i].first.second.coeffs().data(),
                                  pose_vec[i + 1].first.first.data(),
                                  pose_vec[i + 1].first.second.coeffs().data());
        problem->SetManifold(pose_vec[i].first.second.coeffs().data(),
                             quaternion_manifold);
        problem->SetManifold(pose_vec[i + 1].first.second.coeffs().data(),
                             quaternion_manifold);

        // if (j < 5)
        // {
        //     problem->SetParameterBlockConstant(pose_vec[i].first.first.data());
        //     problem->SetParameterBlockConstant(pose_vec[i].first.second.coeffs().data());
        //     j++;
        // }
    }
    // for (int k = 1; k < 2; k++)
    // {
    problem->SetParameterBlockConstant(pose_vec[0].first.first.data());
    problem->SetParameterBlockConstant(pose_vec[0].first.second.coeffs().data());
    // }
    // problem->SetParameterBlockConstant(pose_vec[assFrameInd - sub_frame_num_ + 1].first.first.data());
    // problem->SetParameterBlockConstant(pose_vec[assFrameInd - sub_frame_num_ + 1].first.second.coeffs().data());
    // pose_constant.emplace_back(assFrameInd - sub_frame_num_ + 1);
}

/**
 * @brief 不固定curFrame
 *
 * @param problem
 * @param pose_vec
 * @param curFrameInd
 * @param assFrameInd
 * @param loop_transform
 * @param score
 */
void AddLoopClosureConstrain(ceres::Problem *problem, std::vector<std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>, double>> &pose_vec, const int curFrameInd, const int assFrameInd,
                             const int sub_frame_num_, const std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform, double score)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
    // ceres::LocalParameterization *local_parameter;

    cout << "[Ceres info]:"
         << "current frame index:" << curFrameInd << ","
         << "associated frame index" << assFrameInd << endl;

    /**
     * @brief current frame 和 associated frame的相对位置变换
     *
     */
    ceres::Pose3d pose_c2a;
    pose_c2a.p = loop_transform.first;
    pose_c2a.q = Eigen::Quaterniond(loop_transform.second);

    pose_c2a.q.normalize();

    // problem->AddParameterBlock((pose_vec[curFrameInd].first.first.data()), 3);
    // problem->AddParameterBlock((pose_vec[curFrameInd].first.second.coeffs().data()), 4, local_parameter);
    const Eigen::Matrix<double, 6, 6> sqrt_information = (Eigen::Matrix<double, 6, 6>::Identity() * std_weight).llt().matrixL();
    // const Eigen::Matrix<double, 6, 6> sqrt_information = (Eigen::Matrix<double, 6, 6>::Identity() * std_weight);
    for (int i = 1; i <= sub_frame_num_; i++)
    {
        // if (assFrameInd - i < 0)
        //     continue;

        ceres::CostFunction *cost_function =
            ceres::LoopClosureErrorTerm::Create(pose_c2a, sqrt_information);

        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  pose_vec[assFrameInd + i].first.first.data(),
                                  pose_vec[assFrameInd + i].first.second.coeffs().data(),
                                  pose_vec[curFrameInd + i - sub_frame_num_].first.first.data(),
                                  pose_vec[curFrameInd + i - sub_frame_num_].first.second.coeffs().data());
        problem->SetManifold(pose_vec[assFrameInd + i].first.second.coeffs().data(),
                             quaternion_manifold);
        problem->SetManifold(pose_vec[curFrameInd + i - sub_frame_num_].first.second.coeffs().data(),
                             quaternion_manifold);
    }
}
void AddRelativePoseConstrain(ceres::Problem *problem, const int cloudInd)
{
    // ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LossFunction *loss_function = nullptr;
    ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
    // ceres::LocalParameterization *local_parameter;
    // cout << "[Ceres info]:"
    //      << "current frame index:" << curFrameInd << ","
    //      << "associated frame index" << assFrameInd << endl;
    int j = 0;
    // for (int i = assFrameInd + 1; i < curFrameInd; i++)
    // for (int i = 0; i < curFrameInd; i++)

    ceres::Pose3d curPose, assPose;
    assPose.p = pose_vec[cloudInd - 1].first.first;
    assPose.q = pose_vec[cloudInd - 1].first.second;
    assPose.q = assPose.q.normalized();

    curPose.p = pose_vec[cloudInd].first.first;
    curPose.q = pose_vec[cloudInd].first.second;
    curPose.q = curPose.q.normalized();

    // cout << "[Ceres info]:" << assFrameInd << " " << assPose.p.transpose() << "," << assPose.q.coeffs().transpose() << endl;
    /**
     * @brief current frame 和 associated frame的相对位置变换
     *
     */
    ceres::Pose3d pose_c2a;
    pose_c2a.p = assPose.q.conjugate() * (curPose.p - assPose.p);
    pose_c2a.q = assPose.q.conjugate() * curPose.q;
    pose_c2a.q.normalize();

    const Eigen::Matrix<double, 6, 6> sqrt_information = (Eigen::Matrix<double, 6, 6>::Identity() * lio_weight).llt().matrixL();
    // const Eigen::Matrix<double, 6, 6> sqrt_information = (Eigen::Matrix<double, 6, 6>::Identity() * lio_weight);

    ceres::CostFunction *cost_function =
        ceres::PoseGraph3dErrorTerm::Create(pose_c2a, sqrt_information);

    problem->AddResidualBlock(cost_function,
                              loss_function,
                              pose_vec[cloudInd - 1].first.first.data(),
                              pose_vec[cloudInd - 1].first.second.coeffs().data(),
                              pose_vec[cloudInd].first.first.data(),
                              pose_vec[cloudInd].first.second.coeffs().data());
    problem->SetManifold(pose_vec[cloudInd - 1].first.second.coeffs().data(),
                         quaternion_manifold);
    problem->SetManifold(pose_vec[cloudInd].first.second.coeffs().data(),
                         quaternion_manifold);

    // if (j < 5)
    // {
    //     problem->SetParameterBlockConstant(pose_vec[i].first.first.data());
    //     problem->SetParameterBlockConstant(pose_vec[i].first.second.coeffs().data());
    //     j++;
    // }

    // for (int k = 1; k < 2; k++)
    // {
    if (cloudInd == 1)
    {
        problem->SetParameterBlockConstant(pose_vec[cloudInd - 1].first.first.data());
        problem->SetParameterBlockConstant(pose_vec[cloudInd - 1].first.second.coeffs().data());
    }
    // }
    // problem->SetParameterBlockConstant(pose_vec[assFrameInd - sub_frame_num_ + 1].first.first.data());
    // problem->SetParameterBlockConstant(pose_vec[assFrameInd - sub_frame_num_ + 1].first.second.coeffs().data());
    // pose_constant.emplace_back(assFrameInd - sub_frame_num_ + 1);
}
void SolveOptimizationProblem(ceres::Problem *problem)
{
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.num_threads = 4;

    ceres::Solver::Summary summary;

    auto timeSloevBegin_ = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, problem, &summary);
    auto timeSloevEnd_ = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(timeSloevEnd_ -
                                                                          timeSloevBegin_)
                    .count() *
                1000;
    // problem->SetParameterBlockVariable();
    std::cout << summary.FullReport() << '\n';
    std::cout << "[CeresSolveTime]:" << time << "ms" << endl;

    // for (auto num : pose_constant)
    // {
    //     problem->SetParameterBlockVariable(pose_vec[num].first.first.data());
    //     problem->SetParameterBlockVariable(pose_vec[num].first.second.coeffs().data());
    // }
    // pose_constant.clear();
    // pose_constant.shrink_to_fit();
    // problem->RemoveParameterBlock();
    // problem->RemoveResidualBlock();
}

void SaveTrajOpt(const std::string &traj_file)
{
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open())
    {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : pose_vec)
    {
        ofs << std::fixed << std::setprecision(6) << p.second << " " << std::setprecision(15)
            << p.first.first.x() << " " << p.first.first.y() << " " << p.first.first.z() << " " << p.first.second.x()
            << " " << p.first.second.y() << " " << p.first.second.z() << " " << p.first.second.w() << std::endl;
    }

    ofs.close();
}
void SaveTrajOri(const std::string &traj_file)
{
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open())
    {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : pose_vec_wo_opt)
    {
        ofs << std::fixed << std::setprecision(6) << p.second << " " << std::setprecision(15)
            << p.first.first.x() << " " << p.first.first.y() << " " << p.first.first.z() << " " << p.first.second.x()
            << " " << p.first.second.y() << " " << p.first.second.z() << " " << p.first.second.w() << std::endl;
    }

    ofs.close();
}

void SaveOptPcd()
{
    if (pcl_wait_pub.size() != pose_vec.size())
    {
        ROS_WARN("different size");
    }
    else
    {
        pcl::PointCloud<pcl::PointXYZI> wait_pub;
        int length = pose_vec.size();
        for (int i = 0; i < length; i++)
        {
            // Eigen::Vector3d pv = point2vec(cloud.points[i]);
            // pv = rotation * pv + translation;
            // cloud.points[i] = vec2point(pv);
            Eigen::Vector3d translation = pose_vec[i].first.first;
            Eigen::Quaterniond rotation = pose_vec[i].first.second;
            int ptsize = pcl_wait_pub[i].size();
            for (int j = 0; j < ptsize; j++)
            {
                Eigen::Vector3d pv = point2vec(pcl_wait_pub[i].points[j]);
                pv = rotation * pv + translation;
                pcl_wait_pub[i].points[j] = vec2point(pv);
            }
            wait_pub += pcl_wait_pub[i];
        }
        string file_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST-lio-withSTD/PCD/optscans.pcd");
        pcl::PCDWriter pclSave;
        pclSave.writeBinary(file_name, wait_pub);
    }
}
void SavePclComp(const int curFrameInd, const int assFrameInd, const std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform)
{
    pcl::PointCloud<pcl::PointXYZI> pcl_save;
    pcl::PointCloud<pcl::PointXYZI> optpcl_save;

    // pcl_save.reserve(pcl_wait_pub[curFrameInd].points.size() + pcl_wait_pub[assFrameInd].points.size());

    pcl_save = pcl_wait_pub[assFrameInd];

    for (int ptsize = 0; ptsize < pcl_wait_pub[curFrameInd].points.size(); ptsize++)
    {
        Eigen::Vector3d pv = point2vec(pcl_wait_pub[curFrameInd].points[ptsize]);
        pv = loop_transform.second * pv + loop_transform.first;
        optpcl_save.push_back(vec2point(pv));
    }
    double timeNow = ros::Time::now().toSec();
    auto time1 = to_string(timeNow);

    string file_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST-lio-withSTD/PCD/" + time1 + "_ass.pcd");
    pcl::PCDWriter pclSave;
    pclSave.writeBinary(file_name, pcl_save);

    string optfile_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST-lio-withSTD/PCD/" + time1 + "_cur.pcd");
    pcl::PCDWriter optpclSave;
    optpclSave.writeBinary(optfile_name, optpcl_save);
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "Optimizations");
    ros::NodeHandle nh;
    nh.param<double>("STD/lio_weight", lio_weight, 1.0);
    nh.param<double>("STD/std_weight", std_weight, 1.0);
    nh.param<std::string>("STD/lidar_topic", lidar_topic, "/cloud_registered");
    nh.param<std::string>("STD/odom_topic", odom_topic, "/Odometry");
    nh.param<std::string>("STD/oriTraj_path", oriTraj_path, "/home/crz/Algorithm/FAST-LIO/src/FAST_LIO/Log/oriTran_path.tum");
    nh.param<std::string>("STD/optTraj_path", optTraj_path, "/home/crz/Algorithm/FAST-LIO/src/FAST_LIO/Log/optTran_path.tum");
    /**
     * @brief ROS subscribe initialization
     *
     */
    ros::Subscriber sub_imu = nh.subscribe("/livox/imu", 20000, imu_cbk);
    ros::Subscriber sub_odom = nh.subscribe(odom_topic, 20000, odom_cbk);
    // ros::Subscriber sub_pcl = nh.subscribe("/cloud_registered", 20000, pcl_cbk);
    ros::Subscriber sub_pcl = nh.subscribe(lidar_topic, 2000, pcl_cbk);
    ros::Publisher pubOdomAftMapped =
        nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    ros::Publisher pubCureentCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
    ros::Publisher pubCurrentCorner =
        nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
    ros::Publisher pubMatchedCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
    ros::Publisher pubMatchedCorner =
        nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
    ros::Publisher pubSTD =
        nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

    ros::Publisher pubOdomAftOptted =
        nh.advertise<nav_msgs::Odometry>("/aft_optted", 10);
    /**
     * @brief STD para
     *
     */
    ConfigSetting config_setting;
    read_parameters(nh, config_setting);
    sub_frame_num_ = config_setting.sub_frame_num_;

    STDescManager *std_manager = new STDescManager(config_setting);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    std::vector<double> descriptor_time;
    std::vector<double> querying_time;
    std::vector<double> update_time;
    int triggle_loop_num = 0;

    ros::Rate loop(500);
    ros::Rate slow_loop(10);

    int cloudInd = 0, keyCloudInd = 0, frameInd = 0, loopDetect = 5;
    StatesGroup state_last;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    bool first_state = true;
    ceres::Problem problem;

    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (package_date())
        {
            auto pose_pkg = std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>, double>(data_pkg.first, odom_rcv_time);
            pose_vec_wo_opt.emplace_back(pose_pkg);
            pose_vec.emplace_back(pose_pkg);
            Eigen::Vector3d translation = data_pkg.first.first;
            Eigen::Matrix3d rotation = data_pkg.first.second.toRotationMatrix();
            pcl::PointCloud<pcl::PointXYZI> cloud = *data_pkg.second;
            pcl_wait_pub.emplace_back(*data_pkg.second);

            for (int i = 0; i < cloud.size(); i++)
            {
                Eigen::Vector3d pv = point2vec(cloud.points[i]);
                pv = rotation * pv + translation;
                cloud.points[i] = vec2point(pv);
            }
            down_sampling_voxel(cloud, config_setting.ds_size_);
            for (auto pv : cloud.points)
            {
                temp_cloud->points.push_back(pv);
            }
            if (cloudInd != 0)
            {
                AddRelativePoseConstrain(&problem, cloudInd);
            }

            /**
             * @brief 判断：如果是关键帧，进行PGO，如果不是关键帧，则添加位资节点
             *
             */
            if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0)
            {
                std::cout << "Key Frame id:" << keyCloudInd
                          << ", ckoudInde id:" << cloudInd
                          << ", cloud size: " << temp_cloud->size() << std::endl;

                /**
                 * @brief step1 Descriptor Extraction
                 *
                 */
                auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
                std::vector<STDesc> stds_vec;
                std_manager->GenerateSTDescs(temp_cloud, stds_vec);
                auto t_descriptor_end = std::chrono::high_resolution_clock::now();
                descriptor_time.emplace_back(
                    time_inc(t_descriptor_end, t_descriptor_begin));

                /**
                 * @brief search loop closure
                 *
                 */
                auto t_query_begin = std::chrono::high_resolution_clock::now();
                std::pair<int, double> search_result(-1, 0);
                std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
                loop_transform.first << 0, 0, 0;
                loop_transform.second = Eigen::Matrix3d::Identity();
                std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
                if (keyCloudInd > config_setting.skip_near_num_)
                {
                    std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                            loop_std_pair);
                }
                if (search_result.first > 0)
                {
                    std::cout << "[Loop Detection] triggle loop: " << keyCloudInd
                              << "--" << search_result.first
                              << ", score:" << search_result.second << std::endl;
                    std::cout << "[Loop Transform]" << loop_transform.first.transpose() << "\n"
                              << loop_transform.second << endl;
                }
                auto t_query_end = std::chrono::high_resolution_clock::now();
                querying_time.emplace_back(time_inc(t_query_end, t_query_begin));

                /**
                 * @brief step3 Add descriptors to the database
                 *
                 */
                auto t_map_update_begin = std::chrono::high_resolution_clock::now();
                std_manager->AddSTDescs(stds_vec);
                auto t_map_update_end = std::chrono::high_resolution_clock::now();
                update_time.emplace_back(time_inc(t_map_update_end, t_map_update_begin));
                std::cout << "[Time] descriptor extraction: "
                          << time_inc(t_descriptor_end, t_descriptor_begin) << "ms, "
                          << "query: " << time_inc(t_query_end, t_query_begin)
                          << "ms, "
                          << "update map:"
                          << time_inc(t_map_update_end, t_map_update_begin) << "ms"
                          << std::endl;
                std::cout << std::endl;

                pcl::PointCloud<pcl::PointXYZI> save_key_cloud;
                save_key_cloud = *temp_cloud;

                std_manager->key_cloud_vec_.emplace_back(save_key_cloud.makeShared());

                sensor_msgs::PointCloud2 pub_cloud;
                pcl::toROSMsg(*temp_cloud, pub_cloud);
                pub_cloud.header.frame_id = "camera_init";
                pubCureentCloud.publish(pub_cloud);
                pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
                pub_cloud.header.frame_id = "camera_init";
                pubCurrentCorner.publish(pub_cloud);

                if (search_result.first > 0 && search_result.second > 0.5)
                {
                    std_manager->PlaneGeomrtricIcp(std_manager->plane_cloud_vec_.back(),
                                                   std_manager->plane_cloud_vec_[search_result.first],
                                                   loop_transform);
                    triggle_loop_num++;
                    pcl::toROSMsg(*std_manager->key_cloud_vec_[search_result.first],
                                  pub_cloud);
                    pub_cloud.header.frame_id = "camera_init";
                    pubMatchedCloud.publish(pub_cloud);
                    slow_loop.sleep();
                    pcl::toROSMsg(*std_manager->corner_cloud_vec_[search_result.first],
                                  pub_cloud);
                    pub_cloud.header.frame_id = "camera_init";
                    pubMatchedCorner.publish(pub_cloud);
                    publish_std_pairs(loop_std_pair, pubSTD);
                    slow_loop.sleep();
                    // getchar();
                    /**
                     * @brief add loop closure constrain
                     *
                     */
                    // BuildOptimizationProblem(&problem, pose_vec, cloudInd, search_result.first * config_setting.sub_frame_num_, config_setting.sub_frame_num_);
                    AddLoopClosureConstrain(&problem, pose_vec, cloudInd, search_result.first * config_setting.sub_frame_num_, config_setting.sub_frame_num_, loop_transform, search_result.second);

                    // SavePclComp(keyCloudInd, search_result.first, loop_transform);
                }

                temp_cloud->clear();
                keyCloudInd++;
                loop.sleep();
            }
            SolveOptimizationProblem(&problem);

            nav_msgs::Odometry odom;
            odom.header.frame_id = "camera_init";
            odom.pose.pose.position.x = pose_vec_wo_opt.back().first.first.x();
            odom.pose.pose.position.y = pose_vec_wo_opt.back().first.first.y();
            odom.pose.pose.position.z = pose_vec_wo_opt.back().first.first.z();
            odom.pose.pose.orientation.w = pose_vec_wo_opt.back().first.second.w();
            odom.pose.pose.orientation.x = pose_vec_wo_opt.back().first.second.x();
            odom.pose.pose.orientation.y = pose_vec_wo_opt.back().first.second.y();
            odom.pose.pose.orientation.z = pose_vec_wo_opt.back().first.second.z();
            pubOdomAftMapped.publish(odom);

            nav_msgs::Odometry odom_;
            odom_.header.frame_id = "camera_init";
            odom_.pose.pose.position.x = pose_vec.back().first.first.x();
            odom_.pose.pose.position.y = pose_vec.back().first.first.y();
            odom_.pose.pose.position.z = pose_vec.back().first.first.z();

            odom_.pose.pose.orientation.w = pose_vec.back().first.second.w();
            odom_.pose.pose.orientation.x = pose_vec.back().first.second.x();
            odom_.pose.pose.orientation.y = pose_vec.back().first.second.y();
            odom_.pose.pose.orientation.z = pose_vec.back().first.second.z();
            pubOdomAftOptted.publish(odom_);

            cout << "pose z" << pose_vec.back().first.first.z()
                 << ", pose wo opt z:" << pose_vec_wo_opt.back().first.first.z() << endl;

            loop.sleep();
            cloudInd++;
        }
    }

    SaveTrajOri(oriTraj_path);
    SaveTrajOpt(optTraj_path);
    SaveOptPcd();
    status = ros::ok();
    rate.sleep();

    return 0;
}
