cmake_minimum_required(VERSION 3.18)
project(phot_C)

add_subdirectory(third_party/pybind11)
pybind11_add_module(phot_C phot/csrc/main.cpp)

# DIST_VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
target_compile_definitions(phot_C PRIVATE VERSION_INFO=${DIST_VERSION_INFO})