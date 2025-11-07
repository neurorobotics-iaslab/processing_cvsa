#ifndef PROCESSING_CVSA_HPP_
#define PROCESSING_CVSA_HPP_

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <fftw3.h>
#include <rosneuro_msgs/NeuroOutput.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <rosneuro_buffers_ringbuffer/RingBuffer.h>
#include <processing_cvsa/utils.hpp>
#include <processing_cvsa/features.h>
#include <rosneuro_filters_butterworth/Butterworth.h>

namespace processing{

class CVSA{
public:
    enum class ClassifyResults {BufferNotFull = 0, Error = 1, Success = 2};

public:
    // for ros node
    CVSA(void); // for ros node
    ~CVSA();

    bool configure(void);
    ClassifyResults apply(void);

    void on_received_data(const rosneuro_msgs::NeuroFrame &msg);
    void run(void);
    void set_message(std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> data, std::vector<std::vector<float>> filters_band);

    Eigen::MatrixXcd compute_analytic_signal(const Eigen::MatrixXd& data);

    // for use it as a class
    CVSA(int nchannels, int nsamples, int bufferSize, int filterOrder, int sampleRate, std::string band_str); 
    std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> apply(Eigen::MatrixXf data_in);

protected:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

    rosneuro::Buffer<float>* buffer_;
    bool has_new_data_;
    rosneuro::DynamicMatrix<float> data_in_;
    Eigen::VectorXf rawProb_;
    int nsamples_; // chunk size
    int nchannels_;
    std::vector<std::vector<float>> filters_band_;

    std::vector<uint32_t> idchans_features_;
    Eigen::MatrixXf features_band_;

    std::vector<rosneuro::Butterworth<double>> filters_low_;
    std::vector<rosneuro::Butterworth<double>> filters_high_;

    processing_cvsa::features out_;
    std::string modality_;

    // --- FFTW Members for Hilbert Transform ---
    int fft_buffer_size_; // The size of your buffer (e.g., 512)
    fftw_complex *fft_in_;
    fftw_complex *fft_freq_;
    fftw_complex *fft_out_;
    fftw_plan plan_fwd_;
    fftw_plan plan_bwd_;
};
}

#endif