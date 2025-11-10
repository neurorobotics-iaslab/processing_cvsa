#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <processing_cvsa/eeg_power.h> 
#include "processing_cvsa/utils.hpp" 
#include <string>
#include <vector>


class LoggerNode {
public:
    LoggerNode(ros::NodeHandle& nh) {
        if (!nh.getParam("output_filename", output_filename_)) {
            output_filename_ = "features_output.csv";
            ROS_WARN("Parameter 'output_filename' doesn't found. Default: %s", output_filename_.c_str());
        }

        if (!nh.getParam("band_index", band_index_to_save_)) {
            band_index_to_save_ = 0; 
            ROS_WARN("Parameter 'band_index' dont found. Default: %d", band_index_to_save_);
        } else {
            ROS_INFO("Save band index: %d", band_index_to_save_);
        }

        std::string topic = "/cvsa/eeg_power";
        sub_ = nh.subscribe(topic, 1, &LoggerNode::callback, this);
    }

    ~LoggerNode() {
        if (collected_rows_.empty()) {
            ROS_WARN("No data received.");
            return;
        }

        int n_samples = collected_rows_.size();
        int n_channels = collected_rows_[0].cols(); 

        ROS_INFO("Received %d campioni. Creation final matrix [samples x channels]: (%d x %d)...",
                 n_samples, n_samples, n_channels);

        Eigen::MatrixXf final_matrix(n_samples, n_channels);

        for (int i = 0; i < n_samples; ++i) {
            final_matrix.row(i) = collected_rows_[i];
        }

        writeCSV<float>(output_filename_, final_matrix);
    }


    void callback(const processing_cvsa::eeg_power::ConstPtr& msg) {
        uint32_t n_channels = msg->nchannels;
        if (n_channels == 0) return;

        uint32_t n_bands = msg->data.size() / n_channels;
        if (msg->data.size() % n_channels != 0) {
            ROS_ERROR("Data error: %lu elements non divisible for %u channels!",
                      msg->data.size(), n_channels);
            return;
        }

        if (band_index_to_save_ >= n_bands) {
            ROS_ERROR_ONCE("Error: 'band_index' (%d) out of range. The matrix has only %u bands (indices 0-%u).",
                           band_index_to_save_, n_bands, n_bands - 1);
            return;
        }

        Eigen::Map<const Eigen::MatrixXf> full_matrix(
            msg->data.data(), n_channels, n_bands
        );

        Eigen::VectorXf band_column = full_matrix.col(band_index_to_save_);
        Eigen::RowVectorXf channels_row = band_column.transpose();
        collected_rows_.push_back(channels_row);

        std::cout << "Received sample " << msg->seq << std::endl;
    }

private:
    ros::Subscriber sub_;
    std::string output_filename_;
    int band_index_to_save_;
    
    std::vector<Eigen::RowVectorXf> collected_rows_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_subscriber");
    ros::NodeHandle nh("~"); 

    LoggerNode logger(nh);

    ros::spin(); 

    return 0; 
}