cmake_minimum_required(VERSION 3.18)
project(_C)

add_subdirectory(third_party/pybind11)
pybind11_add_module(_C phot/csrc/main.cpp)

# DIST_VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
target_compile_definitions(_C PRIVATE VERSION_INFO=${DIST_VERSION_INFO})

# This line now should not be added
# set_target_properties(_C PROPERTIES OUTPUT_NAME phot/_C)