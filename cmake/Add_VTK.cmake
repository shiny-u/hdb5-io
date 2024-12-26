message(STATUS "******* FINDING VTK dependency from ${PROJECT_NAME} (minimal requested version: ${VTK_TAG}) *******")
find_package(VTK ${VTK_TAG} REQUIRED NO_MODULE)
message(STATUS "VTK ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION} found on the system here: ${VTK_USE_FILE}")

include(${VTK_USE_FILE})
