/*
    simple test with a single bandpass filter
*/

#include <iostream>
#include <string>
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <rosneuro_buffers_ringbuffer/RingBuffer.hpp>
#include "processing_cvsa/CVSA.hpp"
#include <processing_cvsa/utils.hpp>

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_CVSA_processing");
    std::string datapath = "/home/paolo/cvsa/ic_cvsa_ws/src/processing_cvsa";
    const std::string fileinput  = datapath + "/test/rawdata.csv";
    const std::string fileoutput_pocessing  = datapath + "/test/class/processing.csv";

    // Load input data
    ROS_INFO("Loading data from %s", fileinput.c_str());
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input = readCSV<double>(fileinput);
    int frameSize = 32;
    int nchannels = input.cols();
    int nsamples  = input.rows();
    int bufferSize = 512;
    int size_band_processing = (nsamples/frameSize)-(bufferSize/frameSize - 1);

    // Configure processing
    ROS_INFO("Configure processing");
    int filterOrder = 4;
    float sampleRate = 512.0;
    processing::CVSA cvsa_processor(nchannels, frameSize, bufferSize, filterOrder, sampleRate, "8 14;");

    // Allocate time variables for time analysis
	ros::WallTime start_loop, stop_loop;
    std::vector<double> time_loop;

    // Simulate a rosneuro loop
    Eigen::MatrixXf data;
    Eigen::MatrixXd data_processed = Eigen::MatrixXd::Zero(size_band_processing, nchannels);

    ROS_INFO("Start simulated loop...");
	auto count = 0;
    for(auto i = 0; i<nsamples; i = i+frameSize) {
        start_loop = ros::WallTime::now();

		data = input.middleRows(i, frameSize).cast<float>(); // [samples x channels]
        Eigen::MatrixXd data_p;

        data_p = cvsa_processor.apply(data);
        if(data_p.size() != 0){
            data_processed.middleRows(count, 1) = data_p.col(0).transpose(); // take only the first band
		    count++;
            stop_loop = ros::WallTime::now();

            time_loop.push_back((stop_loop-start_loop).toNSec()/1000.0f);
        }
	}
    ROS_INFO("...simulated loop ended");
    
    Eigen::VectorXd time_loop_eigen = Eigen::Map<Eigen::VectorXd>(time_loop.data(), time_loop.size());
	ROS_INFO("Overall loop time (%d iterations): %f ms", count, time_loop_eigen.mean()); 

    ROS_INFO("Saving data");
    writeCSV<double>(fileoutput_pocessing, data_processed);

    ros::shutdown();
    return 0;
}