file(GLOB nnevaluate_SRC *.cpp *.h *.hpp)
include_directories(
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util)

if (EXISTS ${tensorflow_gpu}/lib/libtensorflow.so)
    message("using tensorflow_c_api GPU")
    include_directories(${tensorflow_gpu}/include)
else() 
    message("using tensorflow_c_api CPU")
    include_directories(${tensorflow_cpu}/include)
endif()

#add_executable(nntest ${nnevaluate_SRC} Test.cxx)
add_library(nnevaluate STATIC ${nnevaluate_SRC})

target_link_libraries(nnevaluate 
        tensorflow
        tensorflow_framework
        pthread
        )

#target_link_libraries(
#        nntest
#        protobuf
#        tensorflow_cc
#        tensorflow_framework
#        pthread
#        hex
#        util
#)
