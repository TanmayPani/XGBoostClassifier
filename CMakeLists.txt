cmake_minimum_required(VERSION 3.18)
project(XGBoostClassifier)

find_package(CUDAToolkit)
find_package(xgboost REQUIRED)

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_executable(${PROJECT_NAME} XGBoostClassifier.cpp)

target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(${PROJECT_NAME} xgboost::xgboost)
target_link_libraries(${PROJECT_NAME} argparse)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)