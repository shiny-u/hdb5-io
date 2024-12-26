    include(FetchContent)

    FetchContent_Declare(meshoui
            GIT_REPOSITORY ${meshoui_URL}
            GIT_TAG ${meshoui_TAG}
            )

    FetchContent_GetProperties(meshoui)
    if(NOT meshoui_POPULATED)
        message(STATUS "******* FETCHING meshoui dependency from ${PROJECT_NAME} (requested version: ${meshoui_TAG}) *******")
        FetchContent_Populate(meshoui)

        # meshoui BUILD OPTIONS
        set(MESHOUI_BUILD_TESTS OFF CACHE BOOL "")
        set(MESHOUI_BUILD_TOOLS OFF CACHE BOOL "")
        set(MESHOUI_USE_VTK ${HDB5IO_USE_VTK} CACHE BOOL "")

        add_subdirectory(${meshoui_SOURCE_DIR} ${meshoui_BINARY_DIR})
    endif()
