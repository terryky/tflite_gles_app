# Make uses /bin/sh by default, which is incompatible with the bashisms seen
# below.
SHELL := /bin/bash

# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

HOST_ARCH := $(shell if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/absl \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(OBJDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target makefiles.
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS := -O3 -DNDEBUG -fPIC
CXXFLAGS += $(EXTRA_CXXFLAGS)
CFLAGS := ${CXXFLAGS}
CXXFLAGS += --std=c++11
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

ifeq ($(HOST_OS),windows)
CXXFLAGS += -fext-numeric-literals -D__LITTLE_ENDIAN__
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a

# Benchmark static library and binary
BENCHMARK_LIB_NAME := benchmark-lib.a
BENCHMARK_BINARY_NAME := benchmark_model
BENCHMARK_PERF_OPTIONS_BINARY_NAME := benchmark_model_performance_options

# A small example program that shows how to link against the library.
MINIMAL_SRCS := \
	tensorflow/lite/examples/minimal/minimal.cc

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

PROFILER_SRCS := \
  tensorflow/lite/profiling/memory_info.cc \
	tensorflow/lite/profiling/time.cc

PROFILE_SUMMARIZER_SRCS := \
	tensorflow/lite/profiling/profile_summarizer.cc \
	tensorflow/core/util/stats_calculator.cc

CMD_LINE_TOOLS_SRCS := \
	tensorflow/lite/tools/command_line_flags.cc

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/lite/*.cc) \
$(wildcard tensorflow/lite/*.c) \
$(wildcard tensorflow/lite/c/*.c) \
$(wildcard tensorflow/lite/core/*.cc) \
$(wildcard tensorflow/lite/core/api/*.cc) \
$(wildcard tensorflow/lite/experimental/resource_variable/*.cc) \
$(wildcard tensorflow/lite/experimental/ruy/*.cc)
ifneq ($(BUILD_TYPE),micro)
CORE_CC_ALL_SRCS += \
$(wildcard tensorflow/lite/kernels/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/optimized/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/reference/*.cc) \
$(PROFILER_SRCS) \
tensorflow/lite/tools/make/downloads/farmhash/src/farmhash.cc \
tensorflow/lite/tools/make/downloads/fft2d/fftsg.c \
tensorflow/lite/tools/make/downloads/fft2d/fftsg2d.c \
tensorflow/lite/tools/make/downloads/flatbuffers/src/util.cpp
CORE_CC_ALL_SRCS += \
	$(shell find tensorflow/lite/tools/make/downloads/absl/absl/ \
	             -type f -name \*.cc | grep -v test | grep -v benchmark | grep -v synchronization | grep -v debugging)
endif
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/lite/*test.cc) \
$(wildcard tensorflow/lite/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/benchmark.cc) \
$(wildcard tensorflow/lite/*/*/example*.cc) \
$(wildcard tensorflow/lite/*/*/test*.cc) \
$(wildcard tensorflow/lite/*/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/lite/kernels/*test_main.cc) \
$(wildcard tensorflow/lite/kernels/*test_util*.cc) \
$(MINIMAL_SRCS)

BUILD_WITH_MMAP ?= true
ifeq ($(BUILD_TYPE),micro)
	BUILD_WITH_MMAP=false
endif
ifeq ($(BUILD_TYPE),windows)
	BUILD_WITH_MMAP=false
endif
ifeq ($(BUILD_WITH_MMAP),true)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/mmap_allocation.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/mmap_allocation_disabled.cc
endif


# -------------------------------------------------------------------------------------------
# for GPU Delegate
# -------------------------------------------------------------------------------------------
CXXFLAGS += -I$(MAKEFILE_DIR)/downloads/fp16/include
CXXFLAGS += -I$(MAKEFILE_DIR)/downloads/flatbuffers/grpc/
CXXFLAGS += -DMESA_EGL_NO_X11_HEADERS
CXXFLAGS += -DEGL_EGLEXT_PROTOTYPES

CORE_CC_ALL_SRCS += tensorflow/lite/c/c_api_internal.c
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_device.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/add_bias.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_go.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_cpp.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_grpc.cpp
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/api2.cc
CORE_CC_ALL_SRCS += external/flatbuffers/grpc/src/compiler/go_generator.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_dart.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_js_ts.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_fbs.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_lobster.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_rust.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_text.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_parser.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/flatc.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/code_generators.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_php.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_lua.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_json_schema.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/grpc/src/compiler/java_generator.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_general.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/reflection.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/flatc_main.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_python.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/grpc/src/compiler/cpp_generator.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/prelu.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/request_gpu_info.cc
CORE_CC_ALL_SRCS += tensorflow/lite/minimal_logging.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/api.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/data_type.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/throw_delegate.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/cycleclock.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/raw_logging.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/spinlock.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/sysinfo.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/thread_identity.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/unscaledcycleclock.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/log_severity.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/dynamic_annotations.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/spinlock_wait.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/types/bad_variant_access.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/opencl_wrapper.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_context.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/str_format/float_conversion.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/str_format/extension.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/convert.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/str_format/output.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/str_format/parser.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/converter.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/relu.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/general_transformations.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/registry.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/memory_management.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/make_fully_connected.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/add.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/reshape.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/memory_management/internal.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/make_padding.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/memory_management/types.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/match_dilated_convolution.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/str_format/arg.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/mul.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/container/internal/hashtablez_sampler.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/remove_noop.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/container/internal/hashtablez_sampler_force_weak_definition.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/debugging/stacktrace.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/conv.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/slice.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/ascii.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/debugging/internal/address_is_readable.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/charconv.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/escaping.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/debugging/internal/elf_mem_image.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/command_queue.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/charconv_bigint.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/charconv_parse.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/debugging/internal/vdso_support.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/memutil.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/match.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/barrier.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/gpu_info.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/workgroups/ideal_workgroup_picker.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/numbers.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/str_cat.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/blocking_counter.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/str_replace.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/str_split.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/internal/create_thread_identity.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/string_view.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/substitute.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/internal/per_thread_sem.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/ostringstream.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/util.cpp
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/internal/waiter.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/float16_conversions.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/softmax.cc
CORE_CC_ALL_SRCS += tensorflow/lite/minimal_logging_default.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/numeric/int128.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/util.cc
## SUBCOMMAND: # @flatbuffers//:flatc [action 'Linking external/flatbuffers/flatc [for host]']
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/notification.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/shape.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/hash/internal/hash.cc
## SUBCOMMAND: # //tensorflow/lite/schema:schema_fbs_srcs [action 'Generating flatbuffer files for schema_fbs_srcs: //tensorflow/lite/schema:schema_fbs_srcs']
## SUBCOMMAND: # //tensorflow/lite/delegates/gpu/cl:compiled_program_cache_cc_fbs_srcs [action 'Generating flatbuffer files for compiled_program_cache_cc_fbs_srcs: //tensorflow/lite/delegates/gpu/cl:compiled_program_cache_cc_fbs_srcs']
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/mutex.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.cc
CORE_CC_ALL_SRCS += tensorflow/lite/util.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/hash/internal/city.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/model_builder.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/types/bad_optional_access.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/synchronization/internal/graphcycles.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_image_format.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_event.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/base/internal/low_level_alloc.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/rename.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/util.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/workgroups/calculator.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_program.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/debugging/symbolize.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/flt_type.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/precision.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/debugging/internal/demangle.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/egl_sync.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/gl_errors.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/civil_time.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/clock.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/container/internal/raw_hash_set.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/transpose_conv.cc
CORE_CC_ALL_SRCS += external/farmhash_archive/src/farmhash.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/code_generators.cpp
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/duration.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_fbs.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_general.cpp
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/format.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_gen_text.cpp
CORE_CC_ALL_SRCS += external/flatbuffers/src/idl_parser.cpp
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/time.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/reflection.cpp
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/civil_time_detail.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/pad.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/utf8.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_memory.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_fixed.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/tensor_type.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_format.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/fuse_inline.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/gl_sync.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/gl_buffer.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_if.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/gl_program.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/gl_shader.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_impl.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/upsampling_bilinear.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/concat.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_libc.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/workgroup_selection.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_lookup.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_posix.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/egl_environment.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/lstm.cc
CORE_CC_ALL_SRCS += external/flatbuffers/src/util.cpp
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/operations.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/model_transformer.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/egl_context.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/types/bad_any_cast.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/program_cache.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/texture2d.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/buffer.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/linear_storage.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/lstm.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/converter.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/environment.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/gl_interop.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/concat_z.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/concat_xy.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/apply_mask.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/add.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/fully_connected_texture.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv_3x3.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_thin.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/selectors/dw_convolution_selector.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_3x3_thin.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/selectors/convolution_transposed_selector.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/conv_powervr.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/conv_constants.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/selectors/fully_connected_selector.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/selectors/convolution_selector.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/time_zone_info.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/egl_surface.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/transpose.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/strided_slice.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/softmax1x1.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/softmax.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/reshapex4.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/reshape.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/relu.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/prelu.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/pooling.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/padding.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/multiply_add.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/max_unpooling.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/tensor.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/cl_kernel.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/gl_texture.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/max_unpooling.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/upsample.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/kernels/elementwise.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/tensor_type_util.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/runtime.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/time/internal/cctz/src/zone_info_source.cc
CORE_CC_ALL_SRCS += tensorflow/lite/kernels/kernel_util.cc
CORE_CC_ALL_SRCS += tensorflow/lite/kernels/internal/quantization_util.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/custom_registry.cc
CORE_CC_ALL_SRCS += external/com_google_absl/absl/strings/internal/str_format/bind.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/selectors/simple_selectors.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/kernels/pooling.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/selectors/operation_selector.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/gl/object_manager.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/inference_context.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/cl/api.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/delegate.cc
CORE_CC_ALL_SRCS += tensorflow/lite/delegates/gpu/delegate.cc
## SUBCOMMAND: # //tensorflow/lite/delegates/gpu:delegate [action 'Linking tensorflow/lite/delegates/gpu/libdelegate.so']
## SUBCOMMAND: # //tensorflow/lite/delegates/gpu:delegate [action 'Linking tensorflow/lite/delegates/gpu/libdelegate.pic.a']
## SUBCOMMAND: # //tensorflow/lite/delegates/gpu:delegate [action 'Linking tensorflow/lite/delegates/gpu/libdelegate.a']



BUILD_WITH_NNAPI ?= true
ifeq ($(BUILD_TYPE),micro)
	BUILD_WITH_NNAPI=false
endif
ifeq ($(TARGET),windows)
	BUILD_WITH_NNAPI=false
endif
ifeq ($(TARGET),ios)
	BUILD_WITH_NNAPI=false
endif
ifeq ($(TARGET),rpi)
	BUILD_WITH_NNAPI=false
endif
ifeq ($(TARGET),generic-aarch64)
	BUILD_WITH_NNAPI=false
endif
ifeq ($(BUILD_WITH_NNAPI),true)
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/nnapi_delegate.cc
  CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_implementation.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_util.cc
	LIBS += -lrt
else
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/nnapi_delegate_disabled.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_implementation_disabled.cc
endif

ifeq ($(TARGET),ios)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_default.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_ios.cc
endif


# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))

# Benchmark sources
BENCHMARK_SRCS_DIR := tensorflow/lite/tools/benchmark
EVALUATION_UTILS_SRCS := \
  tensorflow/lite/tools/evaluation/utils.cc
BENCHMARK_ALL_SRCS := \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*.cc) \
	$(PROFILE_SUMMARIZER_SRCS) \
	$(CMD_LINE_TOOLS_SRCS) \
	$(EVALUATION_UTILS_SRCS)

BENCHMARK_MAIN_SRC := $(BENCHMARK_SRCS_DIR)/benchmark_main.cc
BENCHMARK_PERF_OPTIONS_SRC := \
	$(BENCHMARK_SRCS_DIR)/benchmark_tflite_performance_options_main.cc
BENCHMARK_LIB_SRCS := $(filter-out \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*_test.cc) \
	$(BENCHMARK_MAIN_SRC) \
	$(BENCHMARK_PERF_OPTIONS_SRC) \
	$(BENCHMARK_SRCS_DIR)/benchmark_plus_flex_main.cc, \
	$(BENCHMARK_ALL_SRCS))

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_SRCS := \
	$(MINIMAL_SRCS) \
	$(PROFILER_SRCS) \
	$(PROFILER_SUMMARIZER_SRCS) \
	$(TF_LITE_CC_SRCS) \
	$(BENCHMARK_LIB_SRCS) \
  $(CMD_LINE_TOOLS_SRCS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

LIB_PATH := $(LIBDIR)$(LIB_NAME)
BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)
BENCHMARK_PERF_OPTIONS_BINARY := $(BINDIR)$(BENCHMARK_PERF_OPTIONS_BINARY_NAME)
MINIMAL_BINARY := $(BINDIR)minimal

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

MINIMAL_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MINIMAL_SRCS))))

LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(TF_LITE_CC_SRCS)))))

BENCHMARK_MAIN_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_MAIN_SRC))))

BENCHMARK_PERF_OPTIONS_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_PERF_OPTIONS_SRC))))

BENCHMARK_LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_LIB_SRCS))))

# For normal manually-created TensorFlow Lite C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
# For normal manually-created TensorFlow Lite C source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH)  $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

# The target that's compiled for micro-controllers
micro: $(LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.OPENSOURCE tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): tensorflow/lite/schema/schema_generated.h $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

lib: $(LIB_PATH)

$(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MINIMAL_BINARY) $(MINIMAL_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

minimal: $(MINIMAL_BINARY)

$(BENCHMARK_LIB) : $(LIB_PATH) $(BENCHMARK_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(BENCHMARK_LIB) $(LIB_OBJS) $(BENCHMARK_LIB_OBJS)

benchmark_lib: $(BENCHMARK_LIB)

$(BENCHMARK_BINARY) : $(BENCHMARK_MAIN_OBJ) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_BINARY) $(BENCHMARK_MAIN_OBJ) \
	$(LIBFLAGS) $(BENCHMARK_LIB) $(LDFLAGS) $(LIBS)

$(BENCHMARK_PERF_OPTIONS_BINARY) : $(BENCHMARK_PERF_OPTIONS_OBJ) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_PERF_OPTIONS_BINARY) $(BENCHMARK_PERF_OPTIONS_OBJ) \
	$(LIBFLAGS) $(BENCHMARK_LIB) $(LDFLAGS) $(LIBS)

benchmark: $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

libdir:
	@echo $(LIBDIR)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
