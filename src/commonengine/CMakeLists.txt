file(GLOB commonengine_SRC *.cpp *.hpp)
add_library(commonengine STATIC ${commonengine_SRC})
include_directories(${top_srcdir}/src
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util
        ${top_srcdir}/src/book
        ${top_srcdir}/src/solver
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine
        ${top_srcdir}/src/neuralnet
        )
target_link_libraries(commonengine
        nnevaluate
        hex 
        util 
        book 
        fuego_smartgame
        fuego_gtpengine
        solver
        db
        ${Boost_LIBRARIES})
#        boost_system boost_filesystem boost_program_options boost_thread db)
