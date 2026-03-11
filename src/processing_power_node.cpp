#include <ros/ros.h>
#include "processing_bci/Power.hpp"

int main(int argc, char** argv) {

    
    // ros initialization
    ros::init(argc, argv, "processing_power_node");

    processing::Power power;
    
    if(power.configure() == false) {
        std::cerr<<"[processing_power_node] SETUP ERROR"<<std::endl;
        return -1;
    }

    ROS_INFO("[INFO] Configuration done");
    
    power.run();
    
    ros::shutdown();
    return 0;
}
