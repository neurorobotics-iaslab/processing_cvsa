#ifndef PROCESSING_TEST_CVSA_HPP_
#define PROCESSING_TEST_CVSA_HPP_

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <rosneuro_msgs/NeuroOutput.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <rosneuro_buffers_ringbuffer/RingBuffer.h>
#include <rosneuro_filters_butterworth/Butterworth.h>
#include <processing_cvsa/features.h>
#include <fstream>

namespace processing{

class Test_CVSA{
public:
    enum class ClassifyResults {BufferNotFull = 0, Error = 1, Success = 2};
public:
    Test_CVSA(void);
    ~Test_CVSA();

    bool configure(void);
    ClassifyResults classify(void);

    void on_received_data(const rosneuro_msgs::NeuroFrame &msg);
    void run(void);
    void set_message(std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> data, std::vector<std::vector<float>> filters_band);
    void save_features(std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> data);
    void save_data_filtered(Eigen::MatrixXd data, int idx);
    void save_data_buffer(Eigen::MatrixXd data);


protected:
    bool str2vecOfvec(std::string current_str,  std::vector<std::vector<float>>& out);

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

    rosneuro::Buffer<float>* buffer_;
    bool has_new_data_;
    rosneuro::DynamicMatrix<float> data_in_;
    Eigen::VectorXf rawProb_;
    int nsamples_;
    int nchannels_;
    std::vector<std::vector<float>> filters_band_;

    std::vector<uint32_t> idchans_features_;
    Eigen::MatrixXf features_band_;

    std::vector<rosneuro::Butterworth<double>> filters_low_;
    std::vector<rosneuro::Butterworth<double>> filters_high_;
    rosneuro::Butterworth<double> filter_low_test_;
    rosneuro::Butterworth<double> filter_high_test_;

    processing_cvsa::features out_;

    std::ofstream outputFile_buffer_;
    std::vector<std::ofstream> outputFile_filtered_;
    std::ofstream outputFile_features_; 

    std::string modality_;

};
}

#endif