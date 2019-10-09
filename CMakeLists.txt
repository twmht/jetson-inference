
cmake_minimum_required(VERSION 2.8)
project(jetson-inference)

# submodule warning
# message(" ")
# message("Note:  this project uses git submodules in the source tree.")
# message("       if you haven't already, run the following command from")
# message("       the project's root directory:")
# message(" ")
# message("           git submodule update --init") 
# message("\n")

# if( NOT EXISTS "${PROJECT_SOURCE_DIR}/utils/.git" )
	# message("Note:  required git submodules have not been detected.")
	# message("       first, please run the following command from the")
	# message("       the project's root directory to clone them:")
	# message(" ")
	# message("          git submodule update --init")
	# message(" ")
	# message(FATAL_ERROR "missing required git submodules, see instructions above")
# endif()



# setup tensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")	# -std=gnu++11


# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda" )
message(${CMAKE_MODULE_PATH})
find_package(CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
include("cmake/External/glog.cmake")
# list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
# list(APPEND Caffe_LINKER_LIBS PUBLIC ${GLOG_LIBRARIES})

include_directories(${GLOG_INCLUDE_DIRS})

# setup TensorRT
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}; 
    -O3 
    # -gencode arch=compute_53,code=sm_53
    # -gencode arch=compute_62,code=sm_62

    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
)

# CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		# -gencode arch=compute_35,code=sm_35 \
		# -gencode arch=compute_50,code=sm_50 \
		# -gencode arch=compute_52,code=sm_52 \
		# -gencode arch=compute_60,code=sm_60 \
		# -gencode arch=compute_61,code=sm_61 \
		# -gencode arch=compute_61,code=compute_61

if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

    # set(
        # CUDA_NVCC_FLAGS
        # ${CUDA_NVCC_FLAGS}; 
        # -gencode arch=compute_72,code=sm_72
    # )

    # OpenCV used for findHomography() and decomposeHomography()
    # OpenCV version >= 3.0.0 required for decomposeHomography()
    find_package(OpenCV 3.0.0 COMPONENTS core calib3d REQUIRED PATHS ${OPENCV_ROOT})
endif()

# # setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# # build C/C++ library
include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR}/jetson-inference ${PROJECT_INCLUDE_DIR}/jetson-utils ${PROJECT_INCLUDE_DIR}/plugins)
# TODO: x86-64 and arm64
include_directories(/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/include/glib-2.0 /usr/include/gstreamer-1.0)

include_directories(${TensorRT_INCLUDE_DIRS})
get_filename_component(TENSORT_LIBRARY_DIRS ${TensorRT_LIBRARIES} DIRECTORY)
link_directories(${TENSORT_LIBRARY_DIRS})

file(GLOB inferenceSources c/*.cpp c/*.cu calibration/*.cpp plugins/*.cu plugins/*.cpp)
file(GLOB inferenceIncludes c/*.h c/*.cuh calibration/*.h plugins/*.h plugins/*.hpp)

cuda_add_library(jetson-inference SHARED ${inferenceSources})
#target_link_libraries(jetson-inference nvcaffe_parser nvinfer)		# gstreamer-0.10 gstbase-0.10 gstapp-0.10 


# transfer all headers to the include directory
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/jetson-inference)

foreach(include ${inferenceIncludes})
    message("-- Copying ${include}")
    configure_file(${include} ${PROJECT_INCLUDE_DIR}/jetson-inference COPYONLY)
endforeach()


# create symbolic link for network data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
  
  
# copy image data
file(GLOB imageData ${PROJECT_SOURCE_DIR}/data/images/*)

foreach(image ${imageData})
    message("-- Copying ${image}")
    file(COPY ${image} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    #configure_file(${include} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()


# build subdirectories
add_subdirectory(docs)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(utils)
# add_subdirectory(python)
SET(PY_SOURCE_DIR "python/bindings_v2")
include_directories(pybind11)
add_subdirectory(pybind11)
pybind11_add_module(pyinference ${PY_SOURCE_DIR}/init.cpp ${PY_SOURCE_DIR}/tensornet.cpp ${PY_SOURCE_DIR}/retinaface.cpp)


# set linker options
target_link_libraries(jetson-inference jetson-utils nvinfer nvinfer_plugin nvcaffe_parser ${GLOG_LIBRARIES})

if(CUDA_VERSION_MAJOR GREATER 9)
    target_link_libraries(jetson-inference nvonnxparser opencv_core opencv_calib3d)
endif()


# install includes
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/jetson-inference)
endforeach()

# install symlink to networks
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/networks ${CMAKE_INSTALL_PREFIX}/bin/networks )" )

# install the shared library
install(TARGETS jetson-inference DESTINATION lib EXPORT jetson-inferenceConfig)

# install the cmake project, for importing
install(EXPORT jetson-inferenceConfig DESTINATION share/jetson-inference/cmake)

# run ldconfig after installing
install(CODE "execute_process( COMMAND ldconfig )")

