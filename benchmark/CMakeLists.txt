
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchncnn PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")


add_executable(benchcold benchcold.cpp)
target_link_libraries(benchcold PRIVATE ncnn)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchcold PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchcold PROPERTY FOLDER "benchmark")


add_executable(benchcolddeploy benchcolddeploy.cpp)
target_link_libraries(benchcolddeploy PRIVATE ncnn)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchcolddeploy PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchcolddeploy PROPERTY FOLDER "benchmark")



#add_executable(benchncnn_ benchncnn_.cpp)
#target_link_libraries(benchncnn_ PRIVATE ncnn)
#
#if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
#    target_link_libraries(benchncnn_ PRIVATE nodefs.js)
#endif()
#
## add benchncnn to a virtual project group
#set_property(TARGET benchncnn_ PROPERTY FOLDER "benchmark")
#
#
#
#
#add_executable(benchmark_memory_footprint benchmark_memory_footprint.cpp)
#target_link_libraries(benchmark_memory_footprint PRIVATE ncnn)
#
#if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
#    target_link_libraries(benchmark_memory_footprint PRIVATE nodefs.js)
#endif()
#
## add benchncnn to a virtual project group
#set_property(TARGET benchmark_memory_footprint PROPERTY FOLDER "benchmark")