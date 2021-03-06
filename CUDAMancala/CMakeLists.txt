#GPUMancala project build file

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mancalaCUDA LANGUAGES CXX CUDA)

find_package(Protobuf REQUIRED)
 
add_library(mancala STATIC
           rlagent.cu
           rlagent.cuh
		   Protobuf/TabularRLData.pb.cc
		   Protobuf/TabularRLData.pb.h
  )
  
target_include_directories(mancala PUBLIC ${PROTOBUF_INCLUDE_DIRS})
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(mancala PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( mancala
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#https://devblogs.nvidia.com/building-cuda-applications-cmake/
 set_target_properties(mancala PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(mancala_train train.cu)
 
set_property(TARGET mancala_train 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
			 
target_link_libraries(mancala_train protobuf::libprotobuf)

target_link_libraries(mancala_train mancala CUDA)