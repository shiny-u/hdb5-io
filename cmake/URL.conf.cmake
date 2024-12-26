
# Here we declare the different PATH, TAG and PATCH to get the HDB5_IO dependencies


# MathUtils
set(mathutils_URL git@frydom-ce.org:ce/mathutils.git)
set(mathutils_TAG v1.14.3 CACHE STRING "MathUtils version")
set(MATHUTILS_BUILD_TESTS OFF CACHE BOOL "")
set(MATHUTILS_BUILD_BOOST_TESTS OFF CACHE BOOL "")

# Eigen
set(eigen_URL https://gitlab.com/libeigen/eigen.git  CACHE STRING "eigen repository URL")
set(eigen_TAG 3.3.7 CACHE STRING "eigen version")

# HighFive
set(highfive_URL https://github.com/BlueBrain/HighFive/archive/refs/tags/v2.4.1.tar.gz)
set(highfive_TAG v2.4.1)
set(highfive_PATCH highfive_v2.4.1.patch)

# GoogleTest
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_TAG v1.15.2 CACHE STRING "googletest version")

# meshoui
set(meshoui_URL git@frydom-ce.org:ce/meshoui.git)
set(meshoui_TAG v1.4 CACHE STRING "meshoui version")

# zlib
set(zlib_URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz)
set(zlib_TAG v1.3.1)

# hdf5
set(hdf5_URL https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_13_1.tar.gz)
set(hdf5_TAG 1.13.1)

# VTK
set(VTK_TAG 8.2 CACHE STRING "VTK version")
