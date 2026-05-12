cmake_minimum_required(VERSION 3.10)
project(DMControl VERSION 1.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific settings
if(WIN32)
    add_definitions(-DDM_PLATFORM_WINDOWS)
    set(PLATFORM_LIBS ws2_32)
elseif(UNIX)
    add_definitions(-DDM_PLATFORM_POSIX)
    set(PLATFORM_LIBS pthread)
    if(APPLE)
        add_definitions(-DDM_PLATFORM_MACOS)
    else()
        add_definitions(-DDM_PLATFORM_LINUX)
    endif()
endif()

# Define DM_CONTROL_EXPORTS for DLL
add_definitions(-D DM_CONTROL_EXPORTS)

# Source files
set(DM_CONTROL_SOURCES
    src/dm_control.cpp
    src/controller.cpp
    src/thread_pool.cpp
)

# Headers
set(DM_CONTROL_HEADERS
    include/dm_control.h
    src/controller.h
    src/thread_pool.h
)

# Create shared library
add_library(dm_control SHARED
    ${DM_CONTROL_SOURCES}
    ${DM_CONTROL_HEADERS}
)

# Include directories
target_include_directories(dm_control PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(dm_control PUBLIC ${PLATFORM_LIBS})

# Compiler warnings
if(MSVC)
    target_compile_options(dm_control PRIVATE /W4 /EHsc /MD)
else()
    target_compile_options(dm_control PRIVATE -Wall -Wextra -pedantic)
endif()

# Set output name
set_target_properties(dm_control PROPERTIES
    OUTPUT_NAME "dm_control"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Install targets
install(TARGETS dm_control
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/dm_control.h
    DESTINATION include
)

# Test executable
add_executable(dm_test
    src/test_main.cpp
)

target_link_libraries(dm_test
    dm_control
    ${PLATFORM_LIBS}
)

target_include_directories(dm_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Print configuration
message(STATUS "=========================================")
message(STATUS "  DM Control Configuration")
message(STATUS "=========================================")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "=========================================")