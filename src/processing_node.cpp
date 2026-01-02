#include <ros/ros.h>
#include "processing_bci/CVSA.hpp"

int main(int argc, char** argv) {

    
    // ros initialization
    ros::init(argc, argv, "processing_cvsa_node");

    processing::CVSA cvsa;
    
    if(cvsa.configure() == false) {
        std::cerr<<"[processing_cvsa_node] SETUP ERROR"<<std::endl;
        return -1;
    }

    ROS_INFO("[INFO] Configuration done");
    
    cvsa.run();
    
    ros::shutdown();
    return 0;
}
