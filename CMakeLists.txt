cmake_minimum_required(VERSION 3.14)
project(P3P)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
if (OPENCV_FOUND)
    message("OpenCV Found")
endif()

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    message("Eigen3 Found")
endif()
message("${EIGEN3_INCLUDE_DIR}")
include_directories(include
                    ${EIGEN3_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    )

aux_source_directory(src SOURCE_FILES)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})