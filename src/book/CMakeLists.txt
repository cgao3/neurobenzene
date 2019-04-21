file(GLOB book_SRC *.cpp *.hpp)
include_directories(${top_srcdir}/src
        ${top_srcdir}/src/util
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine)
add_library(book STATIC ${book_SRC})

target_link_libraries(book 
        util
        hex
        fuego_smartgame
        fuego_gtpengine
        pthread
        ${Boost_LIBRARIES}
        )
#boost_system boost_filesystem boost_program_options boost_thread db)

