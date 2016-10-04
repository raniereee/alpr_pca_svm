project("APSVM")

cmake_minimum_required(VERSION 2.6)

set(CMAKE_SYSTEM_NAME Linux)

# Compiladores 
set(CMAKE_CPP_COMPILER g++)

# Habilitar verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

exec_program(pkg-config . ARGS --cflags opencv OUTPUT_VARIABLE OPENCV_FLAGS)
exec_program(pkg-config . ARGS --libs   opencv OUTPUT_VARIABLE OPENCV_LIBS)

# sumir com -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

set(DVC_LIBS            ${OPENCV_LIBS})
set(DVC_FLAGS		${OPENCV_FLAGS})

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${DVC_LIBS}")

add_definitions(-W -Wall -DDEBUG -O0 -g3 ${DVC_FLAGS})

include_directories(./)

set(APSVM_APP  main.cpp)

set(TRAINPCA4SVM trainPCA4SVM.cpp)

add_executable(apsvm ${APSVM_APP})
add_executable(trainPCA4SVM ${TRAINPCA4SVM})
