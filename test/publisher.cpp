#include <ros/ros.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include "processing_bci/utils.hpp"

static const std::vector<std::string> CH_LABELS_32 = {
    "Fp1","Fz","F3","F7","F9","FC5","FC1","C3","T7","P9",
    "CP5","Cp1","Pz","P3","P7","O1","Oz","O2","P4","P8",
    "P10","CP6","CP2","Cz","C4","T8","F10","FC6","FC2","F4","F8","Fp2"
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_publisher");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string csv_filename;
    int n_samples;
    double sample_rate;

    if (!private_nh.getParam("csv_file", csv_filename)) {
        ROS_ERROR("Parameter 'csv_file' not set.");
        return 1;
    }
    if (!private_nh.getParam("chunk_size", n_samples)) {
        ROS_ERROR("Parameter 'chunk_size' not set.");
        return 1;
    }
    if (!private_nh.getParam("sample_rate", sample_rate)) {
        ROS_ERROR("Parameter 'sample_rate' not set.");
        return 1;
    }

    ROS_INFO("Loading data from: %s", csv_filename.c_str());
    Eigen::MatrixXd full_data;
    try {
        full_data = readCSV<double>(csv_filename);
    } catch (const std::exception& e) {
        ROS_ERROR("Error reading CSV: %s", e.what());
        return 1;
    }

    int n_channels    = full_data.cols();
    int total_samples = full_data.rows();
    ROS_INFO("Loaded: %d samples x %d channels.", total_samples, n_channels);

    // Build label list (truncate or extend to match actual channel count)
    std::vector<std::string> ch_labels(CH_LABELS_32.begin(),
                                       CH_LABELS_32.begin() + std::min(n_channels, (int)CH_LABELS_32.size()));
    for (int i = (int)ch_labels.size(); i < n_channels; i++)
        ch_labels.push_back("Ch" + std::to_string(i + 1));

    ros::Publisher pub = nh.advertise<rosneuro_msgs::NeuroFrame>("/neurodata", 1);
    ros::Rate loop_rate(sample_rate / n_samples);

    ROS_INFO("Waiting for subscriber on /neurodata ...");
    while (ros::ok() && pub.getNumSubscribers() == 0) {
        ros::Duration(0.5).sleep();
        ROS_INFO_THROTTLE(5.0, "Still waiting...");
    }
    ROS_INFO("Subscriber connected. Starting publication.");

    int current_sample = 0;
    uint32_t seq = 0;
    while (ros::ok()) {
        if (current_sample + n_samples > total_samples) {
            ROS_INFO("End of CSV file.");
            break;
        }

        Eigen::MatrixXd chunk = full_data.block(current_sample, 0, n_samples, n_channels);

        rosneuro_msgs::NeuroFrame msg;
        msg.header.stamp    = ros::Time::now();
        msg.header.seq      = seq;
        msg.sr              = sample_rate;
        msg.eeg.info.nchannels = n_channels;
        msg.eeg.info.nsamples  = n_samples;
        msg.eeg.info.labels    = ch_labels;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> chunk_float = chunk.cast<float>();
        msg.eeg.data.assign(chunk_float.data(), chunk_float.data() + chunk_float.size());

        pub.publish(msg);
        ROS_INFO_THROTTLE(1.0, "Published seq %u", seq);

        current_sample += n_samples;
        seq++;

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
