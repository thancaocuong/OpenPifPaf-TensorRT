cmake_minimum_required(VERSION 3.6)
project(openpifpaf LANGUAGES CXX)
find_package(CUDA REQUIRED)
find_cuda_helper_libs(npp)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_INSTALL_DIR /usr/local/cuda)

set(TARGET_NAME openpifpaf)

file(GLOB HEADER_LIST
        "${OpenPifPaf_SOURCE_DIR}/include/openpifpaf/*.hpp")

file(GLOB SOURCE_LIST
        "${OpenPifPaf_SOURCE_DIR}/src/*.cpp"
        "${OpenPifPaf_SOURCE_DIR}/src/*.cu")


cuda_add_library(${TARGET_NAME} STATIC ${SOURCE_LIST} ${HEADER_LIST})
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(openpifpaf
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV)

find_library(CUDART_LIBRARY cudart nppig nppicc nppc HINTS /usr/local/cuda/lib64)

target_include_directories(${TARGET_NAME}
    PRIVATE stdc++fs
    PUBLIC ../include
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    )

set(SAMPLE_DEP_LIBS
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
    ${OpenCV_LIBS}
    ${CUDART_LIBRARY}
    ${CUDA_npp_LIBRARY}
    nvinfer
)
add_definitions(-O2 -pthread)
target_link_libraries(openpifpaf ${SAMPLE_DEP_LIBS})
target_compile_options(openpifpaf PUBLIC -Wreturn-local-addr -Wdeprecated-declarations -Wnarrowing)
