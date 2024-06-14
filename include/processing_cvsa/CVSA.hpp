#ifndef PROCESSING_CVSA_HPP_
#define PROCESSING_CVSA_HPP_

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <rosneuro_msgs/NeuroOutput.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <rosneuro_buffers_ringbuffer/RingBuffer.hpp>
#include <rosneuro_decoder/Decoder.h>

namespace processing{

class CVSA{
public:
    CVSA(void);
    ~CVSA();

    bool configure(void);
    bool classify(void);

    void on_received_data(const rosneuro_msgs::NeuroFrame &msg);
    void run(void);
    void set_message(void);

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    rosneuro_msgs::NeuroOutput out_;

    rosneuro::Buffer<float>* buffer_;
    rosneuro::decoder::Decoder* decoder_;
    bool has_new_data_;
    rosneuro::DynamicMatrix<float> data_in_;
    Eigen::VectorXf rawProb_;
    int nsamples_;
    int nchannels_;

    std::ofstream outputFile_;

private:
    //void butterworthBandpass(int order, double lowcut, double highcut, double sampleRate, std::vector<double>& a, std::vector<double>& b);
    //Eigen::MatrixXd applyFilterToMatrix(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::MatrixXd& data);
    //Eigen::VectorXd applyFilter(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::VectorXd& data);
    std::vector<double> TrinomialMultiply(int FilterOrder, std::vector<double> b, std::vector<double> c);
std::vector<double> ComputeDenCoeffs(int FilterOrder, double Lcutoff, double Ucutoff);
std::vector<double> ComputeNumCoeffs(int FilterOrder, double Lcutoff, double Ucutoff, std::vector<double> DenC);
std::vector<double> ComputeHP(int FilterOrder);
std::vector<double> ComputeLP(int FilterOrder);
Eigen::MatrixXf filter(const Eigen::MatrixXf& in, const std::vector<double>& b, const std::vector<double>& a);
};
}

#endif