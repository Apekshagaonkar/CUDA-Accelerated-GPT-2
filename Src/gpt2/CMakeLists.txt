include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable( attention
${HEADERS}
attention.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
attention
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${CUDA_CUBLAS_LIBRARIES}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries( 
attention
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${CUDA_CUBLAS_LIBRARIES}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)

cuda_add_executable( layernorm
${HEADERS}
layernorm.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
layernorm
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${CUDA_CUBLAS_LIBRARIES}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries( 
layernorm
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${CUDA_CUBLAS_LIBRARIES}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)

cuda_add_executable( softmax
${HEADERS}
softmax.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
softmax
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${CUDA_CUBLAS_LIBRARIES}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries( 
softmax
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${CUDA_CUBLAS_LIBRARIES}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)