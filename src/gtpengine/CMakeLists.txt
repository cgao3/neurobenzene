file(GLOB gtpengine_SRC *.cpp *.h *.hpp)

include_directories(${top_srcdir}/src)

add_library(fuego_gtpengine STATIC ${gtpengine_SRC})

target_link_libraries(fuego_gtpengine
        ${Boost_LIBRARIES} )
#boost_system boost_program_options boost_filesystem boost_thread)
