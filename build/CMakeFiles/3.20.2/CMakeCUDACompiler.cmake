set(CMAKE_CUDA_COMPILER "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.6.124")
set(CMAKE_CUDA_DEVICE_LINKER "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "8.5")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/targets/x86_64-linux/lib/stubs;/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/include;/usr/include/c++/8;/usr/include/c++/8/x86_64-redhat-linux;/usr/include/c++/8/backward;/usr/lib/gcc/x86_64-redhat-linux/8/include;/usr/local/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/targets/x86_64-linux/lib/stubs;/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/targets/x86_64-linux/lib;/usr/lib/gcc/x86_64-redhat-linux/8;/usr/lib64;/lib64;/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/lib64;/usr/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
