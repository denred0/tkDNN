# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vid/tkDNN/tkDNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vid/tkDNN/tkDNN/build

# Include any dependencies generated for this target.
include CMakeFiles/map_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/map_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/map_demo.dir/flags.make

CMakeFiles/map_demo.dir/demo/demo/map.cpp.o: CMakeFiles/map_demo.dir/flags.make
CMakeFiles/map_demo.dir/demo/demo/map.cpp.o: ../demo/demo/map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vid/tkDNN/tkDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/map_demo.dir/demo/demo/map.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/map_demo.dir/demo/demo/map.cpp.o -c /home/vid/tkDNN/tkDNN/demo/demo/map.cpp

CMakeFiles/map_demo.dir/demo/demo/map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/map_demo.dir/demo/demo/map.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vid/tkDNN/tkDNN/demo/demo/map.cpp > CMakeFiles/map_demo.dir/demo/demo/map.cpp.i

CMakeFiles/map_demo.dir/demo/demo/map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/map_demo.dir/demo/demo/map.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vid/tkDNN/tkDNN/demo/demo/map.cpp -o CMakeFiles/map_demo.dir/demo/demo/map.cpp.s

# Object files for target map_demo
map_demo_OBJECTS = \
"CMakeFiles/map_demo.dir/demo/demo/map.cpp.o"

# External object files for target map_demo
map_demo_EXTERNAL_OBJECTS =

map_demo: CMakeFiles/map_demo.dir/demo/demo/map.cpp.o
map_demo: CMakeFiles/map_demo.dir/build.make
map_demo: libtkDNN.so
map_demo: libkernels.so
map_demo: /usr/local/cuda/lib64/libcudadevrt.a
map_demo: /usr/local/cuda/lib64/libcudart_static.a
map_demo: /usr/lib/x86_64-linux-gnu/libcudnn.so
map_demo: /home/vid/tensorrt/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib/libnvinfer.so
map_demo: /usr/local/lib/libopencv_gapi.so.4.5.1
map_demo: /usr/local/lib/libopencv_stitching.so.4.5.1
map_demo: /usr/local/lib/libopencv_alphamat.so.4.5.1
map_demo: /usr/local/lib/libopencv_aruco.so.4.5.1
map_demo: /usr/local/lib/libopencv_bgsegm.so.4.5.1
map_demo: /usr/local/lib/libopencv_bioinspired.so.4.5.1
map_demo: /usr/local/lib/libopencv_ccalib.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudabgsegm.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudafeatures2d.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudaobjdetect.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudastereo.so.4.5.1
map_demo: /usr/local/lib/libopencv_cvv.so.4.5.1
map_demo: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.1
map_demo: /usr/local/lib/libopencv_dnn_superres.so.4.5.1
map_demo: /usr/local/lib/libopencv_dpm.so.4.5.1
map_demo: /usr/local/lib/libopencv_face.so.4.5.1
map_demo: /usr/local/lib/libopencv_freetype.so.4.5.1
map_demo: /usr/local/lib/libopencv_fuzzy.so.4.5.1
map_demo: /usr/local/lib/libopencv_hfs.so.4.5.1
map_demo: /usr/local/lib/libopencv_img_hash.so.4.5.1
map_demo: /usr/local/lib/libopencv_intensity_transform.so.4.5.1
map_demo: /usr/local/lib/libopencv_line_descriptor.so.4.5.1
map_demo: /usr/local/lib/libopencv_mcc.so.4.5.1
map_demo: /usr/local/lib/libopencv_quality.so.4.5.1
map_demo: /usr/local/lib/libopencv_rapid.so.4.5.1
map_demo: /usr/local/lib/libopencv_reg.so.4.5.1
map_demo: /usr/local/lib/libopencv_rgbd.so.4.5.1
map_demo: /usr/local/lib/libopencv_saliency.so.4.5.1
map_demo: /usr/local/lib/libopencv_stereo.so.4.5.1
map_demo: /usr/local/lib/libopencv_structured_light.so.4.5.1
map_demo: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.1
map_demo: /usr/local/lib/libopencv_superres.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudacodec.so.4.5.1
map_demo: /usr/local/lib/libopencv_surface_matching.so.4.5.1
map_demo: /usr/local/lib/libopencv_tracking.so.4.5.1
map_demo: /usr/local/lib/libopencv_highgui.so.4.5.1
map_demo: /usr/local/lib/libopencv_datasets.so.4.5.1
map_demo: /usr/local/lib/libopencv_plot.so.4.5.1
map_demo: /usr/local/lib/libopencv_text.so.4.5.1
map_demo: /usr/local/lib/libopencv_videostab.so.4.5.1
map_demo: /usr/local/lib/libopencv_videoio.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudaoptflow.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudalegacy.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudawarping.so.4.5.1
map_demo: /usr/local/lib/libopencv_optflow.so.4.5.1
map_demo: /usr/local/lib/libopencv_xfeatures2d.so.4.5.1
map_demo: /usr/local/lib/libopencv_ml.so.4.5.1
map_demo: /usr/local/lib/libopencv_shape.so.4.5.1
map_demo: /usr/local/lib/libopencv_ximgproc.so.4.5.1
map_demo: /usr/local/lib/libopencv_video.so.4.5.1
map_demo: /usr/local/lib/libopencv_dnn.so.4.5.1
map_demo: /usr/local/lib/libopencv_xobjdetect.so.4.5.1
map_demo: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
map_demo: /usr/local/lib/libopencv_objdetect.so.4.5.1
map_demo: /usr/local/lib/libopencv_calib3d.so.4.5.1
map_demo: /usr/local/lib/libopencv_features2d.so.4.5.1
map_demo: /usr/local/lib/libopencv_flann.so.4.5.1
map_demo: /usr/local/lib/libopencv_xphoto.so.4.5.1
map_demo: /usr/local/lib/libopencv_photo.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudaimgproc.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudafilters.so.4.5.1
map_demo: /usr/local/lib/libopencv_imgproc.so.4.5.1
map_demo: /usr/local/lib/libopencv_cudaarithm.so.4.5.1
map_demo: /usr/local/lib/libopencv_core.so.4.5.1
map_demo: /usr/lib/x86_64-linux-gnu/librt.so
map_demo: /usr/local/cuda/lib64/libcublas.so
map_demo: /usr/local/lib/libopencv_cudev.so.4.5.1
map_demo: /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.6.2
map_demo: CMakeFiles/map_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vid/tkDNN/tkDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable map_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/map_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/map_demo.dir/build: map_demo

.PHONY : CMakeFiles/map_demo.dir/build

CMakeFiles/map_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/map_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/map_demo.dir/clean

CMakeFiles/map_demo.dir/depend:
	cd /home/vid/tkDNN/tkDNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vid/tkDNN/tkDNN /home/vid/tkDNN/tkDNN /home/vid/tkDNN/tkDNN/build /home/vid/tkDNN/tkDNN/build /home/vid/tkDNN/tkDNN/build/CMakeFiles/map_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/map_demo.dir/depend

