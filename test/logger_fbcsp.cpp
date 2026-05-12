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
        uint32_t n_components = msg->ncomponents_for_band;
        if (n_components == 0) return;

        uint32_t n_bands = msg->nbands;
        uint32_t total_features = n_components * n_bands;

        if (msg->data.size() != total_features) {
            ROS_ERROR("Data error: size %lu does not match components * bands!",
                      msg->data.size());
            return;
        }

        Eigen::Map<const Eigen::RowVectorXf> components_row(
            msg->data.data(), 
            total_features
        );

        // Aggiungiamo la riga mappata al nostro std::vector
        collected_rows_.push_back(components_row);

        std::cout << "Received sample " << msg->seq << std::endl;
    }

private:
    ros::Subscriber sub_;
    std::string output_filename_;
    
    std::vector<Eigen::RowVectorXf> collected_rows_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_subscriber_fbcsp");
    ros::NodeHandle nh("~"); 

    LoggerNode logger(nh);

    ros::spin(); 

    return 0; 
}
