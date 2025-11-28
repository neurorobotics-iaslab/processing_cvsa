#ifndef PROCESSING_CVSA_HPP_
#define PROCESSING_CVSA_HPP_

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <fftw3.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <rosneuro_buffers_ringbuffer/RingBuffer.h>
#include <processing_cvsa/utils.hpp>
#include <processing_cvsa/eeg_power.h>
#include <rosneuro_filters_butterworth/Butterworth.h>

namespace processing{

class CVSA{
public:
    enum class ApplyResults {BufferNotFull = 0, Error = 1, Success = 2};

public:
    // for ros node
    CVSA(void); 
    ~CVSA();

    bool configure(void);
    ApplyResults apply(void);

    void on_received_data(const rosneuro_msgs::NeuroFrame &msg);
    void run(void);
    void set_message(Eigen::MatrixXd data);

    Eigen::MatrixXcd compute_analytic_signal(const Eigen::MatrixXd& data);

    // for use it as a class
    CVSA(int nchannels, int frameSize, int bufferSize, int filterOrder, int sampleRate, std::string band_str); 
    Eigen::MatrixXd apply(Eigen::MatrixXf data_in);

protected:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

    std::vector<rosneuro::Buffer<float>*> buffers_;
    bool has_new_data_;
    rosneuro::DynamicMatrix<float> data_in_;
    int chunkSize_; // chunk size
    int nchannels_;
    int nfilters_;
    std::vector<std::vector<float>> filters_band_;
    int seq_id_;
    bool is_configured_;

    std::vector<rosneuro::Butterworth<double>> filters_low_;
    std::vector<rosneuro::Butterworth<double>> filters_high_;

    processing_cvsa::eeg_power out_;
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