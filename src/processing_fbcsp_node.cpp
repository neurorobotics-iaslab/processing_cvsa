#include <ros/ros.h>
#include "processing_bci/Fbcsp.hpp"

int main(int argc, char** argv) {

    // ros initialization
    ros::init(argc, argv, "processing_fbcsp_node");

    processing::Fbcsp fbcsp;
    
    if(fbcsp.configure() == false) {
        ROS_ERROR("[FBCSP Processing] Configuration failed");
        ros::shutdown();
        return 0;
    }
    
    fbcsp.run();
    
    ros::shutdown();
    return 0;
}
