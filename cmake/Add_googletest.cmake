include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY ${googletest_URL}
        GIT_TAG ${googletest_TAG}
        )

FetchContent_GetProperties(googletest)

if (NOT googletest_POPULATED)
    message(STATUS "******* FETCHING googletest dependency from ${PROJECT_NAME} (requested version: ${googletest_TAG}) *******")
    FetchContent_Populate(googletest)

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "")

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(
            ${googletest_SOURCE_DIR}
            ${googletest_BINARY_DIR}
            EXCLUDE_FROM_ALL)
endif ()

