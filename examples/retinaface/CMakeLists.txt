
file(GLOB retinafaceSources *.cpp)
file(GLOB retinafaceIncludes *.h )

cuda_add_executable(retinaface ${retinafaceSources})

target_link_libraries(retinaface jetson-inference)

install(TARGETS retinaface DESTINATION bin)
