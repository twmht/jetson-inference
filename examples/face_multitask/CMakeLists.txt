
file(GLOB multiSources *.cpp)
file(GLOB multiIncludes *.h )

cuda_add_executable(face_multitask ${multiSources})

target_link_libraries(face_multitask jetson-inference)

install(TARGETS face_multitask DESTINATION bin)
