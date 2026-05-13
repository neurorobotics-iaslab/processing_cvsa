#ifndef FBCSP_HPP_
#define FBCSP_HPP_

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <rosneuro_msgs/NeuroFrame.h>
#include <rosneuro_buffers_ringbuffer/RingBuffer.h>
#include <processing_bci/utils.hpp>
#include <processing_bci/eeg_fbcsp.h>
#include <rosneuro_filters_butterworth/Butterworth.h>
#include "rosneuro_filters_car/Car.h"

namespace processing {

class Fbcsp {
public:
    enum class ApplyResults {BufferNotFull = 0, Error = 1, Success = 2};

public:
    Fbcsp(void); 
    ~Fbcsp();

    bool configure(void);
    ApplyResults apply(void);

    void on_received_data(const rosneuro_msgs::NeuroFrame &msg);
    void run(void);
    void set_message(const Eigen::MatrixXd& data);

protected:
    bool load_matrix(const std::string& param_name, Eigen::MatrixXd& mat);
    bool load_matrices(const std::string& param_name, std::vector<Eigen::MatrixXd>& mats);

protected:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

    std::string name_;

    std::vector<rosneuro::Buffer<float>*> buffers_;
    bool has_new_data_;
    rosneuro::DynamicMatrix<float> data_in_;
    int chunkSize_; // chunk size
    int nchannels_;
    int nfilters_;
    std::vector<std::vector<float>> filters_band_;
    int seq_id_;
    bool is_configured_;

    rosneuro::Car<double> car_filter_;
    bool do_car_;
    bool is_car_configured_;
    std::vector<std::string> EOG_ch_names_; // channel names from car.yaml
    std::vector<int> EOG_ch_;               // resolved 0-based indices

    std::vector<std::string> csp_ch_names_; // channel names from CspCfg (subset for CSP)
    std::vector<int> csp_ch_idx_;           // resolved 0-based indices into NeuroFrame channels
    bool csp_ch_idx_resolved_;

    std::vector<rosneuro::Butterworth<double>> filters_low_;
    std::vector<rosneuro::Butterworth<double>> filters_high_;

    processing_bci::eeg_fbcsp out_;
    std::string run_mode_, signal_type_;

    // FBCSP and ICA
    bool do_ica_;
    Eigen::MatrixXd ica_matrix_;
    std::vector<Eigen::MatrixXd> csp_matrices_;
    int ncomponents_;
};

}

#endif
