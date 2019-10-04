file(GLOB retinafaceSources *.cpp)
file(GLOB retinafaceIncludes *.h )

cuda_add_executable(sampleint8 ${retinafaceSources})

target_link_libraries(sampleint8 jetson-inference)

install(TARGETS sampleint8 DESTINATION bin)
