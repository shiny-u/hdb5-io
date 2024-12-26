
add_executable(test_import_export test_import_export.cpp)
target_link_libraries(test_import_export hdb5_io gtest gtest_main)
set_target_properties(test_import_export PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)

if (HDB5IO_USE_VTK)
    target_compile_definitions(test_import_export PUBLIC -DUSE_VTK)
endif ()

add_test(NAME test_import_export COMMAND test_import_export)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../data/Boxbarge_Vertices_353_Faces_652.hdb5"
        DESTINATION "${CMAKE_BINARY_DIR}/bin/tests")

add_executable(test_import test_import.cpp)
target_link_libraries(test_import hdb5_io)

if (HDB5IO_USE_VTK)
    target_compile_definitions(test_import PUBLIC -DUSE_VTK)
endif ()
set_target_properties(test_import PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)
