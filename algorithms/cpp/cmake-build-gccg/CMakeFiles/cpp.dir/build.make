# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zy/.ssh/Monoid-Group/algorithms/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg

# Include any dependencies generated for this target.
include CMakeFiles/cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpp.dir/flags.make

CMakeFiles/cpp.dir/main.cpp.o: CMakeFiles/cpp.dir/flags.make
CMakeFiles/cpp.dir/main.cpp.o: ../main.cpp
CMakeFiles/cpp.dir/main.cpp.o: CMakeFiles/cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cpp.dir/main.cpp.o"
	/opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cpp.dir/main.cpp.o -MF CMakeFiles/cpp.dir/main.cpp.o.d -o CMakeFiles/cpp.dir/main.cpp.o -c /Users/zy/.ssh/Monoid-Group/algorithms/cpp/main.cpp

CMakeFiles/cpp.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpp.dir/main.cpp.i"
	/opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zy/.ssh/Monoid-Group/algorithms/cpp/main.cpp > CMakeFiles/cpp.dir/main.cpp.i

CMakeFiles/cpp.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpp.dir/main.cpp.s"
	/opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zy/.ssh/Monoid-Group/algorithms/cpp/main.cpp -o CMakeFiles/cpp.dir/main.cpp.s

CMakeFiles/cpp.dir/Ssort.cpp.o: CMakeFiles/cpp.dir/flags.make
CMakeFiles/cpp.dir/Ssort.cpp.o: ../Ssort.cpp
CMakeFiles/cpp.dir/Ssort.cpp.o: CMakeFiles/cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cpp.dir/Ssort.cpp.o"
	/opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cpp.dir/Ssort.cpp.o -MF CMakeFiles/cpp.dir/Ssort.cpp.o.d -o CMakeFiles/cpp.dir/Ssort.cpp.o -c /Users/zy/.ssh/Monoid-Group/algorithms/cpp/Ssort.cpp

CMakeFiles/cpp.dir/Ssort.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpp.dir/Ssort.cpp.i"
	/opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zy/.ssh/Monoid-Group/algorithms/cpp/Ssort.cpp > CMakeFiles/cpp.dir/Ssort.cpp.i

CMakeFiles/cpp.dir/Ssort.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpp.dir/Ssort.cpp.s"
	/opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zy/.ssh/Monoid-Group/algorithms/cpp/Ssort.cpp -o CMakeFiles/cpp.dir/Ssort.cpp.s

# Object files for target cpp
cpp_OBJECTS = \
"CMakeFiles/cpp.dir/main.cpp.o" \
"CMakeFiles/cpp.dir/Ssort.cpp.o"

# External object files for target cpp
cpp_EXTERNAL_OBJECTS =

../bin/cpp: CMakeFiles/cpp.dir/main.cpp.o
../bin/cpp: CMakeFiles/cpp.dir/Ssort.cpp.o
../bin/cpp: CMakeFiles/cpp.dir/build.make
../bin/cpp: CMakeFiles/cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpp.dir/build: ../bin/cpp
.PHONY : CMakeFiles/cpp.dir/build

CMakeFiles/cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpp.dir/clean

CMakeFiles/cpp.dir/depend:
	cd /Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zy/.ssh/Monoid-Group/algorithms/cpp /Users/zy/.ssh/Monoid-Group/algorithms/cpp /Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg /Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg /Users/zy/.ssh/Monoid-Group/algorithms/cpp/cmake-build-gccg/CMakeFiles/cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cpp.dir/depend

