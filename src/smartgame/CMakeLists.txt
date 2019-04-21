file(GLOB smartgame_SRC *.cpp *.h *.hpp)

add_library(fuego_smartgame STATIC ${smartgame_SRC})
include_directories(
        ${top_srcdir}/src
        ${top_srcdir}/src/gtpengine/)

target_link_libraries(fuego_smartgame
        ${Boost_LIBRARIES}
        )
#boost_system boost_filesystem boost_program_options boost_thread)
