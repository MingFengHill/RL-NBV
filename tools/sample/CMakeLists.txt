cmake_minimum_required(VERSION 3.3...3.12 FATAL_ERROR)

project(mesh_sampling)

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(VTK 8.2 REQUIRED)
include(${VTK_USE_FILE})

add_executable (mesh_sampling mesh_sampling.cpp)
target_link_libraries (mesh_sampling ${PCL_LIBRARIES} ${VTK_LIBRARIES})
