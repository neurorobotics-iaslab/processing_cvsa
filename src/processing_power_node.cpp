#include <ros/ros.h>
#include "processing_bci/Power.hpp"

int main(int argc, char** argv) {

    
    // ros initialization
    ros::init(argc, argv, "processing_power_node");

    processing::Power power;
    
    if(power.configure() == false) {
        ROS_ERROR("[Power Processing] Configuration failed");
        ros::shutdown();
        return 0;
    }
    
    power.run();
    
    ros::shutdown();
    return 0;
}
