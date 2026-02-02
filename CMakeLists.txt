cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES native)
project(CudaHelloWorld LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") # Enable debugging symbols & GPU debugging
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create executable
add_executable(${PROJECT_NAME} src/main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
