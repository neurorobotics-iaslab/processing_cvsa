cmake_minimum_required(VERSION 3.0.2)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project(processing_cvsa)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosneuro_msgs
  rosneuro_filters
  rosneuro_buffers
  rosneuro_buffers_ringbuffer
  rosneuro_filters_butterworth
  std_msgs
  message_generation
) 
find_package(PkgConfig)

find_package(Eigen3 REQUIRED)
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
)

# online CVSA library
add_library(lib_CVSA_online
                          src/CVSA.cpp
                          )

target_link_libraries(lib_CVSA_online
                    ${Eigen3_LIBRARIES} 
									  ${catkin_LIBRARIES})

add_dependencies(lib_CVSA_online
				 ${catkin_EXPORTED_TARGETS}
)

# test CVSA library
add_library(lib_CVSA_test
                          test/Test_CVSA.cpp
                          ) 
target_link_libraries(lib_CVSA_test
                          ${Eigen3_LIBRARIES} 
                          ${catkin_LIBRARIES})
      
add_dependencies(lib_CVSA_test
               ${catkin_EXPORTED_TARGETS}
)


################ EXEC
add_executable(processing_cvsa_node src/processing_node.cpp) # online
target_link_libraries(processing_cvsa_node 
                          lib_CVSA_online
                          ${RTF_LIBRARIES}
                          ) # fftw3 

add_executable(test_CVSA_node test/test_CVSA_node.cpp) # test all
target_link_libraries(test_CVSA_node 
lib_CVSA_test
                          ${RTF_LIBRARIES}
                          )

add_executable(test_CVSA_processing test/test_CVSA_processing.cpp)
target_link_libraries(test_CVSA_processing 
                        lib_CVSA_online
                        ${RTF_LIBRARIES}
                      )  
