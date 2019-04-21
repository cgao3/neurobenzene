file(GLOB solver_SRC *.cpp *.hpp)
add_library(solver STATIC ${solver_SRC})
include_directories(${top_srcdir}/src
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/book
        ${top_srcdir}/src/util
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine
        ${top_srcdir}/src/neuralnet
        )

target_link_libraries(solver
        ${CMAKE_BINARY_DIR}/src/hex/libhex.a 
        ${CMAKE_BINARY_DIR}/src/util/libutil.a
        book
        fuego_smartgame
        fuego_gtpengine
        nnevaluate
        pthread
        db
        ${Boost_LIBRARIES}
        )
#boost_date_time boost_system boost_filesystem boost_program_options boost_thread db)
