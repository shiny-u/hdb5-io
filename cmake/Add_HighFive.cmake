include(FetchContent)

FetchContent_Declare(highfive
        URL ${highfive_URL}
        PATCH_COMMAND git init && git apply --verbose --ignore-whitespace "${PROJECT_SOURCE_DIR}/cmake/patches/${highfive_PATCH}"
)

set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_EIGEN ON CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_OPENCV OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_XTENSOR OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_PARALLEL_HDF5 OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_INSTALL_DEPS ON CACHE BOOL "" FORCE)

message(STATUS "******* FETCHING highfive dependency from ${PROJECT_NAME} (requested version: ${highfive_TAG}) *******")
FetchContent_MakeAvailable(highfive)
