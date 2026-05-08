#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <processing_bci/eeg_fbcsp.h> 
#include "processing_bci/utils.hpp" 
#include <string>
#include <vector>

class LoggerNode {
public:
    LoggerNode(ros::NodeHandle& nh) {
        if (!nh.getParam("output_filename", output_filename_)) {
            output_filename_ = "fbcsp_features_output.csv";
            ROS_WARN("Parameter 'output_filename' doesn't found. Default: %s", output_filename_.c_str());
        }

        if (!nh.getParam("band_index", band_index_to_save_)) {
            band_index_to_save_ = 0; 
            ROS_WARN("Parameter 'band_index' dont found. Default: %d", band_index_to_save_);
        } else {
            ROS_INFO("Save band index: %d", band_index_to_save_);
        }

        std::string topic;
        nh.param<std::string>("topic", topic, "/eeg_fbcsp");
        sub_ = nh.subscribe(topic, 1, &LoggerNode::callback, this);
    }

    ~LoggerNode() {
        if (collected_rows_.empty()) {
            ROS_WARN("No data received.");
            return;
        }

        int n_samples = collected_rows_.size();
        int n_components = collected_rows_[0].cols(); 

        ROS_INFO("Received %d campioni. Creation final matrix [samples x components]: (%d x %d)...",
                 n_samples, n_samples, n_components);

        Eigen::MatrixXf final_matrix(n_samples, n_components);

        for (int i = 0; i < n_samples; ++i) {
            final_matrix.row(i) = collected_rows_[i];
        }

        writeCSV<float>(output_filename_, final_matrix);
    }

    void callback(const processing_bci::eeg_fbcsp::ConstPtr& msg) {
        uint32_t n_components = msg->ncomponents;
        if (n_components == 0) return;

        uint32_t n_bands = msg->nbands;
        if (msg->data.size() != n_components * n_bands) {
            ROS_ERROR("Data error: size %lu does not match components * bands!",
                      msg->data.size());
            return;
        }

        if (band_index_to_save_ >= n_bands) {
            ROS_ERROR_ONCE("Error: 'band_index' (%d) out of range. The matrix has only %u bands (indices 0-%u).",
                           band_index_to_save_, n_bands, n_bands - 1);
            return;
        }

        Eigen::Map<const Eigen::MatrixXf> full_matrix(
            msg->data.data(), n_components, n_bands
        );

        Eigen::VectorXf band_column = full_matrix.col(band_index_to_save_);
        Eigen::RowVectorXf components_row = band_column.transpose();
        collected_rows_.push_back(components_row);

        std::cout << "Received sample " << msg->seq << std::endl;
    }

private:
    ros::Subscriber sub_;
    std::string output_filename_;
    int band_index_to_save_;
    
    std::vector<Eigen::RowVectorXf> collected_rows_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_subscriber_fbcsp");
    ros::NodeHandle nh("~"); 

    LoggerNode logger(nh);

    ros::spin(); 

    return 0; 
}
