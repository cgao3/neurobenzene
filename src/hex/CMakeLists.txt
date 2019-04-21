file(GLOB hex_SRC *.cpp *.hpp)

include_directories(${top_srcdir}/src
        ${top_srcdir}/src/util
        ${top_srcdir}/src/book
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine)
add_library(hex STATIC ${hex_SRC})
target_link_libraries(hex 
        util 
        book
        fuego_smartgame 
        fuego_gtpengine
        db
        ${Boost_LIBRARIES}
        )
#boost_system boost_filesystem boost_program_options boost_thread)
