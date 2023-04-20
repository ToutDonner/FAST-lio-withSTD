#include <ros/ros.h>
#include <ros/time.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
// #include <cstdio>
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

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "common_lib.h"
#include "STDesc.h"

// #include "Preintegration.h"

std::mutex mtx_buffer;
std::condition_variable sig_buffer;

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
std::vector<std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>,
                      double>>
    pose_vec_with_pgo;

std::vector<pcl::PointCloud<pcl::PointXYZI>> pcl_wait_pub;

gtsam::Values results;

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
        string file_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST_LIO/PCD/optscans.pcd");
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

    string file_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST_LIO/PCD/" + time1 + "_ass.pcd");
    pcl::PCDWriter pclSave;
    pclSave.writeBinary(file_name, pcl_save);

    string optfile_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST_LIO/PCD/" + time1 + "_cur.pcd");
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

    ros::Publisher pubCorrectCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
    ros::Publisher pubOdomCorreted =
        nh.advertise<nav_msgs::Odometry>("/odom_corrected", 10);
    /**
     * @brief STD para
     *
     */
    ConfigSetting config_setting;
    read_parameters(nh, config_setting);
    STDescManager *std_manager = new STDescManager(config_setting);

    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
        gtsam::noiseModel::Diagonal::Variances(Vector6);
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
    double loopNoiseScore = 0.1;
    gtsam::Vector robustNoiseVector6(
        6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
        loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1),
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    std::vector<double> descriptor_time;
    std::vector<double> querying_time;
    std::vector<double> update_time;
    int triggle_loop_num = 0;

    ros::Rate loop(500);
    ros::Rate slow_loop(10);

    int cloudInd = 0, keyCloudInd = 0, frameInd = 0;
    StatesGroup state_last;
    ceres::Problem problem;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    bool first_state = true;

    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (package_date())
        {
            auto pose_pkg = std::pair<std::pair<Eigen::Vector3d, Eigen::Quaterniond>, double>(data_pkg.first, odom_rcv_time);
            pose_vec_wo_opt.push_back(pose_pkg);
            pose_vec.push_back(pose_pkg);
            Eigen::Vector3d translation = data_pkg.first.first;
            Eigen::Matrix3d rotation = data_pkg.first.second.toRotationMatrix();
            pcl::PointCloud<pcl::PointXYZI> cloud = *data_pkg.second;
            pcl_wait_pub.push_back(*data_pkg.second);

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
            /**
             * @brief 添加节点
             *
             */
            if (cloudInd == 0)
            {
                initial.insert(0,
                               gtsam::Pose3(gtsam::Rot3(pose_vec[cloudInd].first.second),
                                            gtsam::Point3(pose_vec[cloudInd].first.first)));
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                    0,
                    gtsam::Pose3(gtsam::Rot3(pose_vec[cloudInd].first.second),
                                 gtsam::Point3(pose_vec[cloudInd].first.first)),
                    odometryNoise));
            }
            else
            {
                // add connection between near frame
                initial.insert(cloudInd,
                               gtsam::Pose3(gtsam::Rot3(pose_vec[cloudInd].first.second),
                                            gtsam::Point3(pose_vec[cloudInd].first.first)));
                Eigen::Vector3d t_ab = pose_vec[cloudInd - 1].first.first;
                Eigen::Matrix3d R_ab = pose_vec[cloudInd - 1].first.second.toRotationMatrix();

                t_ab = R_ab.transpose() * (pose_vec[cloudInd].first.first - t_ab);
                R_ab = R_ab.transpose() * pose_vec[cloudInd].first.second.toRotationMatrix();

                gtsam::Rot3 R_sam(R_ab);
                gtsam::Point3 t_sam(t_ab);

                gtsam::NonlinearFactor::shared_ptr near_factor(
                    new gtsam::BetweenFactor<gtsam::Pose3>(cloudInd - 1, cloudInd,
                                                           gtsam::Pose3(R_sam, t_sam),
                                                           odometryNoise));
                graph.push_back(near_factor);
            }

            /**
             * @brief 判断：如果是关键帧，进行PGO，如果不是关键帧，则添加位资节点
             *
             */
            if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0)
            {
                std::cout << "Key Frame id:" << keyCloudInd
                          << ", cloud size: " << temp_cloud->size() << std::endl;

                /**
                 * @brief step1 Descriptor Extraction
                 *
                 */
                auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
                std::vector<STDesc> stds_vec;
                std_manager->GenerateSTDescs(temp_cloud, stds_vec);
                auto t_descriptor_end = std::chrono::high_resolution_clock::now();
                descriptor_time.push_back(
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
                querying_time.push_back(time_inc(t_query_end, t_query_begin));

                /**
                 * @brief step3 Add descriptors to the database
                 *
                 */
                auto t_map_update_begin = std::chrono::high_resolution_clock::now();
                std_manager->AddSTDescs(stds_vec);
                auto t_map_update_end = std::chrono::high_resolution_clock::now();
                update_time.push_back(time_inc(t_map_update_end, t_map_update_begin));
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

                std_manager->key_cloud_vec_.push_back(save_key_cloud.makeShared());

                sensor_msgs::PointCloud2 pub_cloud;
                pcl::toROSMsg(*temp_cloud, pub_cloud);
                pub_cloud.header.frame_id = "camera_init";
                pubCureentCloud.publish(pub_cloud);
                pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
                pub_cloud.header.frame_id = "camera_init";
                pubCurrentCorner.publish(pub_cloud);

                if (search_result.first > 0 && search_result.second > 0.5)
                {
                    int match_frame = search_result.first;

                    std_manager->PlaneGeomrtricIcp(std_manager->plane_cloud_vec_.back(),
                                                   std_manager->plane_cloud_vec_[match_frame], loop_transform);

                    // source i
                    // target i-1

                    int sub_frame_num = config_setting.sub_frame_num_;
                    for (size_t j = 1; j <= sub_frame_num; j++)
                    {
                        int src_frame = cloudInd + j - sub_frame_num;
                        Eigen::Matrix3d src_R =
                            loop_transform.second * pose_vec[src_frame].first.second.toRotationMatrix();
                        Eigen::Vector3d src_t =
                            loop_transform.second * pose_vec[src_frame].first.first +
                            loop_transform.first;
                        int tar_frame = match_frame * sub_frame_num + j;
                        Eigen::Matrix3d tar_R = pose_vec[tar_frame].first.second.toRotationMatrix();
                        Eigen::Vector3d tar_t = pose_vec[tar_frame].first.first;

                        gtsam::Point3 ttem(tar_R.transpose() * (src_t - tar_t));
                        gtsam::Rot3 Rtem(tar_R.transpose() * src_R);
                        gtsam::NonlinearFactor::shared_ptr loop_factor(
                            new gtsam::BetweenFactor<gtsam::Pose3>(tar_frame, src_frame,
                                                                   gtsam::Pose3(Rtem, ttem),
                                                                   robustLoopNoise));
                        graph.push_back(loop_factor);
                    }

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

                    auto t_pgo_begin = std::chrono::high_resolution_clock::now();
                    gtsam::ISAM2Params parameters;
                    parameters.relinearizeThreshold = 0.01;
                    parameters.relinearizeSkip = 1;
                    gtsam::ISAM2 isam(parameters);
                    isam.update(graph, initial);
                    isam.update();
                    auto t_pgo_end = std::chrono::high_resolution_clock::now();
                    std::cout << "Solve pgo time:" << time_inc(t_pgo_end, t_pgo_begin) << " ms"
                              << std::endl;
                    results = isam.calculateEstimate();
                }

                temp_cloud->clear();
                keyCloudInd++;
                loop.sleep();
            }

            nav_msgs::Odometry odom;
            odom.header.frame_id = "camera_init";
            odom.pose.pose.position.x = translation[0];
            odom.pose.pose.position.y = translation[1];
            odom.pose.pose.position.z = translation[2];
            Eigen::Quaterniond q(rotation);
            odom.pose.pose.orientation.w = q.w();
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            pubOdomAftMapped.publish(odom);
            loop.sleep();
            cloudInd++;
        }
    }

    SaveTrajOri(oriTraj_path);
    if (1)
    {
        pcl::PointCloud<pcl::PointXYZI> wait_pub;
        for (int i = 0; i < results.size(); i++)
        {
            // Eigen::Vector3d pv = point2vec(cloud.points[i]);
            // pv = rotation * pv + translation;
            // cloud.points[i] = vec2point(pv);
            gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();

            Eigen::Vector3d translation = pose.translation();
            Eigen::Quaterniond rotation(pose.rotation().matrix());
            int ptsize = pcl_wait_pub[i].size();
            for (int j = 0; j < ptsize; j++)
            {
                Eigen::Vector3d pv = point2vec(pcl_wait_pub[i].points[j]);
                pv = rotation * pv + translation;
                pcl_wait_pub[i].points[j] = vec2point(pv);
                pcl_wait_pub[i].points[j].intensity = 125;
            }
            wait_pub += pcl_wait_pub[i];
        }
        string file_name = string("/home/crz/Algorithm/FAST-LIO/src/FAST_LIO/PCD/optscans.pcd");
        pcl::PCDWriter pclSave;
        pclSave.writeBinary(file_name, wait_pub);
    }
    std::ofstream ofs;
    ofs.open(optTraj_path, std::ios::out);
    if (!ofs.is_open())
    {
        LOG(ERROR) << "Failed to open traj_file: " << optTraj_path;
    }
    else
    {
        ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;

        for (int i = 0; i < results.size(); i++)
        {
            gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();

            Eigen::Vector3d opt_translation = pose.translation();
            Eigen::Quaterniond opt_q(pose.rotation().matrix());
            ofs << std::fixed << std::setprecision(6) << pose_vec[i].second << " " << std::setprecision(15)
                << opt_translation.x() << " " << opt_translation.y() << " " << opt_translation.z() << " " << opt_q.x()
                << " " << opt_q.y() << " " << opt_q.z() << " " << opt_q.w() << std::endl;
        }
        ofs.close();
    }

    status = ros::ok();
    rate.sleep();

    return 0;
}
