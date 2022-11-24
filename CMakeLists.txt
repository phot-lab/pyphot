cmake_minimum_required(VERSION 3.4...3.18)
project(phot_C)

add_subdirectory(third_party/pybind11)
pybind11_add_module(phot_C phot/csrc/main.cpp)
