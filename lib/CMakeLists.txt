add_subdirectory( utilities )
add_subdirectory( options )
add_subdirectory( logging )
add_subdirectory( dio )
add_subdirectory( dsp )

# Allow includes like "utilities/time.hpp" when compiling targets in lib/.
target_include_directories(dio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# disruptor library
add_library( disruptor disruptor/claim_strategy.cpp disruptor/wait_strategy.cpp disruptor/sequence.cpp)


