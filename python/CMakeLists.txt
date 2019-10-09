
#
# This list contains the versions of Python that will be detected 
# and the bindings built against (if that version is installed).
#
# If you want to support another version of Python, add it here.
#
# set(PYTHON_BINDING_VERSIONS 3.7.4)

# message("-- trying to build Python bindings for Python versions:  ${PYTHON_BINDING_VERSIONS}")

# foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
	# add_subdirectory(bindings bindings_python_${PYTHON_BINDING_VERSION})
# endforeach()


# #
# # install Python examples
# #
# file(GLOB pyInferenceExamples examples/*.py)

# foreach(pyExample ${pyInferenceExamples})
	# message("-- Copying ${pyExample}")
	# file(COPY ${pyExample} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	# install(FILES "${pyExample}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
# endforeach()

# SET(SOURCE_DIR "bindings_v2")
# include_directories(${SOURCE_DIR})
# include_directories(../pybind11)
# add_subdirectory(../pybind11)
# pybind11_add_module(pyinference ${SOURCE_DIR}/init.cpp ${SOURCE_DIR}/tensornet.cpp)
