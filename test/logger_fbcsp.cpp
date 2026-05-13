#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <processing_bci/eeg_fbcsp.h>
#include "processing_bci/utils.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

class LoggerNode {
public:
    LoggerNode(ros::NodeHandle& nh) {
        if (!nh.getParam("output_filename", output_filename_)) {
            output_filename_ = "fbcsp_features_output.csv";
            ROS_WARN("[Logger] 'output_filename' not set. Default: %s", output_filename_.c_str());
        }
        first_seq_filename_ = output_filename_.substr(0, output_filename_.rfind('.')) + "_first_seq.txt";

        std::string topic;
        nh.param<std::string>("topic", topic, "/eeg_fbcsp");
        sub_ = nh.subscribe(topic, 10, &LoggerNode::callback, this);

        ROS_INFO("[Logger] Saving FBCSP features to: %s", output_filename_.c_str());
        ROS_INFO("[Logger] Subscribing to: %s", topic.c_str());
    }

    ~LoggerNode() {
        if (features_.empty()) {
            ROS_WARN("[Logger] No data received.");
            return;
        }

        uint32_t n_rows = max_seq_ + 1;
        Eigen::MatrixXf out = Eigen::MatrixXf::Zero(n_rows, n_features_);
        for (uint32_t i = 0; i < n_rows; i++) {
            if (i < features_.size() && !features_[i].empty())
                out.row(i) = Eigen::Map<Eigen::RowVectorXf>(features_[i].data(), n_features_);
        }

        writeCSV<float>(output_filename_, out);
        ROS_INFO("[Logger] Saved %u frames (%u features each) to %s",
                 n_rows, n_features_, output_filename_.c_str());

        std::ofstream fs(first_seq_filename_);
        if (fs.is_open()) {
            fs << first_seq_ << "\n";
            fs.close();
            ROS_INFO("[Logger] First seq: %u → saved to %s", first_seq_, first_seq_filename_.c_str());
        } else {
            ROS_ERROR("[Logger] Could not write first_seq file: %s", first_seq_filename_.c_str());
        }
    }

    void callback(const processing_bci::eeg_fbcsp::ConstPtr& msg) {
        uint32_t seq = msg->seq;

        if (first_call_) {
            first_seq_ = seq;
            first_call_ = false;
            n_features_ = msg->ncomponents_for_band * msg->nbands;
            ROS_INFO("[Logger] First seq: %u  |  features/frame: %u", first_seq_, n_features_);
        }

        if (n_features_ == 0 || msg->data.size() != n_features_) {
            ROS_ERROR_THROTTLE(5.0, "[Logger] Feature size mismatch: got %zu, expected %u",
                               msg->data.size(), n_features_);
            return;
        }

        if (seq >= features_.size())
            features_.resize(seq + 1);

        features_[seq].assign(msg->data.begin(), msg->data.end());
        max_seq_ = std::max(max_seq_, seq);

        ROS_INFO_THROTTLE(2.0, "[Logger] Received seq %u", seq);
    }

private:
    ros::Subscriber sub_;
    std::string output_filename_;
    std::string first_seq_filename_;

    std::vector<std::vector<float>> features_;
    uint32_t n_features_ = 0;
    uint32_t max_seq_    = 0;
    uint32_t first_seq_  = 0;
    bool first_call_     = true;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_logger_fbcsp");
    ros::NodeHandle nh("~");

    LoggerNode logger(nh);
    ros::spin();

    return 0;
}
