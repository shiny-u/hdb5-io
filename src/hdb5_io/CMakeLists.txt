
if (HDB5IO_BUILD_SHARED_LIB)
    add_library(hdb5_io SHARED)
else ()
    add_library(hdb5_io STATIC)
endif ()

target_sources(hdb5_io PRIVATE
        containers/HydrodynamicDataBase.cpp
        containers/Body.cpp
        containers/Mask.cpp
        containers/WaveDrift.cpp
        containers/PoleResidue.cpp
        containers/Kochin.cpp
        io/HDBReader.cpp
        io/HDBWriter.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
        )

if (HDB5IO_MESH_SUPPORT)
    target_sources(hdb5_io PRIVATE
            containers/Mesh.cpp
            )
endif ()

target_include_directories(hdb5_io PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(hdb5_io PUBLIC
        MathUtils::MathUtils
        eigen
        HighFive
        hdf5_cpp-static
        )

if (HDB5IO_MESH_SUPPORT)
    target_link_libraries(hdb5_io PUBLIC meshoui)
endif()

add_dependencies(hdb5_io check_git_${PROJECT_NAME}) # For git_watcher to fetch git informations before effective build

if (HDB5IO_USE_VTK)
    target_link_libraries(hdb5_io PUBLIC ${VTK_LIBRARIES})
    target_compile_definitions(hdb5_io PUBLIC -DOMPI_SKIP_MPICXX)
    target_compile_definitions(hdb5_io PUBLIC -DUSE_VTK)
endif ()

if (HDB5IO_MESH_SUPPORT)
    target_compile_definitions(hdb5_io PUBLIC MESH_SUPPORT)
endif ()

set_target_properties(hdb5_io PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
