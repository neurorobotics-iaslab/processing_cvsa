cmake_minimum_required(VERSION 3.0.2)
project(processing_cvsa)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosneuro_msgs
  Eigen3
  rosneuro_filters
  rosneuro_buffers
  rosneuro_buffers_ringbuffer
  rosneuro_filters_butterworth
  std_msgs
  message_generation
) 

find_library(RTF_LIBRARIES rtfilter)
find_path(RTF_INCLUDE_DIRS rtfilter.h)

add_message_files(
  FILES
  features.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS 
    include
  LIBRARIES 
    ${PROJECT_NAME}
	
  CATKIN_DEPENDS
  	roscpp
	  std_msgs
    rosneuro_filters
    rosneuro_buffers
    rosneuro_buffers_ringbuffer
    rosneuro_filters_butterworth
    rosneuro_msgs
    message_runtime
  DEPENDS
)



###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${RTF_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} 
                          src/CVSA.cpp
                          #test/Test_CVSA.cpp
                          )

target_link_libraries(${PROJECT_NAME} 
                    ${Eigen3_LIBRARIES} 
									  ${catkin_LIBRARIES})

add_dependencies(${PROJECT_NAME} 
				 ${catkin_EXPORTED_TARGETS}
)

add_executable(processing_cvsa_node src/processing_node.cpp)
target_link_libraries(processing_cvsa_node 
                          ${catkin_LIBRARIES} 
                          ${PROJECT_NAME}
                          ${RTF_LIBRARIES}
                          ) # fftw3

#add_executable(test_CVSA_node test/test_CVSA_node.cpp)
#target_link_libraries(test_CVSA_node 
#                          ${catkin_LIBRARIES} 
#                          ${PROJECT_NAME}
#                          ${RTF_LIBRARIES}
#                          )

add_executable(test_CVSA_processing test/test_CVSA_processing.cpp)
target_link_libraries(test_CVSA_processing 
                          ${catkin_LIBRARIES} 
                          ${PROJECT_NAME}
                          ${RTF_LIBRARIES}
                      )  
