
file(GLOB faceidSources *.cpp)
file(GLOB faceidIncludes *.h )

cuda_add_executable(faceid ${faceidSources})

target_link_libraries(faceid jetson-inference)

install(TARGETS faceid DESTINATION bin)
