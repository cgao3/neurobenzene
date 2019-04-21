file(GLOB simpleplyers_SRC *.cpp *.hpp)
add_library(simpleplayers STATIC ${simpleplyers_SRC})
include_directories(${top_srcdir}/src
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util
        ${top_srcdir}/src/book
        ${top_srcdir}/src/solver
        ${top_srcdir}/src/commonengine
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine
        ${top_srcdir}/src/neuralnet
        )
target_link_libraries(simpleplayers
        nnevaluate
        hex
        book
       fuego_smartgame
       fuego_gtpengine
       util solver commonengine
       db
       ${Boost_LIBRARIES})
#boost_system boost_filesystem boost_thread boost_program_options db)
