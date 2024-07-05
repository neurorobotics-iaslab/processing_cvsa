#include <iostream>
#include <string>
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <rosneuro_filters_butterworth/Butterworth.hpp>
#include <rosneuro_buffers_ringbuffer/RingBuffer.hpp>
#include <processing_cvsa/utils.hpp>

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_CVSA_processing");
    std::string datapath = "/home/paolo/cvsa_ws/src/processing_cvsa";
    const std::string fileinput  = datapath + "/test/rawdata.csv";
    const std::string fileoutput_bandpass  = datapath + "/test/bandpass.csv";
    const std::string fileoutput_pow  = datapath + "/test/pow.csv";
    const std::string fileoutput_avg  = datapath + "/test/avg.csv";
    const std::string fileoutput_log  = datapath + "/test/log.csv";
    const std::string fileoutput_buffer  = datapath + "/test/buffer.csv";

    // Load input data
    ROS_INFO("Loading data from %s", fileinput.c_str());
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input = readCSV<double>(fileinput);
    int frameSize = 32;
    int nchannels = input.cols();
    int nsamples  = input.rows();
    int bufferSize = 512;
    int size_band_pow = ((nsamples/frameSize)-(bufferSize/frameSize - 1))*bufferSize;
    int size_avg_log = (nsamples/frameSize) - (bufferSize/frameSize - 1);

    // Configure filters
    ROS_INFO("Configure filters");
    int filterOrder = 4;
    float sampleRate = 512.0;
    std::vector<double> band = {8.0, 14.0};
    rosneuro::Butterworth<double> low_pass(rosneuro::ButterType::LowPass,  filterOrder,  band.at(1), sampleRate);
    rosneuro::Butterworth<double> high_pass(rosneuro::ButterType::HighPass,  filterOrder,  band.at(0), sampleRate);
    rosneuro::Buffer<double>* buffer = new rosneuro::RingBuffer<double>();
    buffer->configure("RingBufferCfg");

    // Allocate time variables for time analysis
	ros::WallTime start_loop, stop_loop;
    std::vector<double> time_loop;

    // Simulate a rosneuro loop
    rosneuro::DynamicMatrix<double> data;
    Eigen::MatrixXd buffer_data = Eigen::MatrixXd::Zero(size_band_pow, nchannels);
    rosneuro::DynamicMatrix<double> bandpass = rosneuro::DynamicMatrix<double>::Zero(size_band_pow, nchannels);
    Eigen::MatrixXd pow = Eigen::MatrixXd::Zero(size_band_pow, nchannels);
    Eigen::MatrixXd avg = Eigen::MatrixXd::Zero(size_avg_log, nchannels);
    Eigen::MatrixXd log = Eigen::MatrixXd::Zero(size_avg_log, nchannels);

    ROS_INFO("Start simulated loop");
	auto count = 0;
    for(auto i = 0; i<nsamples; i = i+frameSize) {

        buffer->add(input.middleRows(i, frameSize));
        if(!buffer->isfull()) {
            ROS_WARN("Buffer not full");
            continue;
        }
        start_loop = ros::WallTime::now();

		data = buffer->get();
        buffer_data.middleRows(count*bufferSize, bufferSize) = data;
        Eigen::MatrixXd data_low, data_band, data_pow, data_avg, data_log;
		
        ROS_INFO("Applying bandpass filter");
		data_low = low_pass.apply(data);
        data_band = high_pass.apply(data_low);
        bandpass.middleRows(count*bufferSize, bufferSize) = data_band;

        ROS_INFO("Applying power");
        data_pow = data_band.array().pow(2);
        pow.middleRows(count*bufferSize, bufferSize) = data_pow;

        ROS_INFO("Applying average");
        data_avg = data_pow.colwise().mean();
        avg.middleRows(count, 1) = data_avg;

        ROS_INFO("Applying log");
        data_log = data_avg.array().log();
        log.middleRows(count, 1) = data_log;

		count++;
        stop_loop = ros::WallTime::now();

        time_loop.push_back((stop_loop-start_loop).toNSec()/1000.0f);
	}
    
    Eigen::VectorXd time_loop_eigen = Eigen::Map<Eigen::VectorXd>(time_loop.data(), time_loop.size());
	
	ROS_INFO("Loop ended: filters applied on data");
	ROS_INFO("Overall loop time (%d iterations): %f ms", count, time_loop_eigen.mean()); 
    

    ROS_INFO("Saving data");
    writeCSV<double>(fileoutput_buffer, buffer_data);
    writeCSV<double>(fileoutput_bandpass, bandpass);
    writeCSV<double>(fileoutput_pow, pow);
    writeCSV<double>(fileoutput_avg, avg);
    writeCSV<double>(fileoutput_log, log);

    ros::shutdown();
    return 0;
}