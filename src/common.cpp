#include "common.h"

namespace jaffe {

	static shared_ptr<Jaffe> m_thread_instance;

	Jaffe& Jaffe::Get() {
		if (!m_thread_instance.get()) {
			m_thread_instance.reset(new Jaffe());
		}
		return *(m_thread_instance.get());
	}

#ifdef CPU_ONLY
	Jaffe::Jaffe()
		: m_mode(Jaffe::CPU) {}

	Jaffe::~Jaffe() {}	

	void Jaffe::SetDevice(const int device_id) {
		NO_GPU;
	}

	void Jaffe::DeviceQuery() {
		NO_GPU;
	}

	bool Jaffe::CheckDevice(const int device_id) {
		NO_GPU;
		return false;
	}

	int Jaffe::FindDevice(const int start_id) {
		NO_GPU;
		return -1;
	}

#else
	Jaffe::Jaffe()
		: m_cublas_handle(NULL), m_mode(Jaffe::CPU) {
		if (cublasCreate(&m_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
			std::cout << __FILE__ << "\t" << __LINE__ << " ERROR: Cannot create " <<
				"Cublas handl. Cublas won't be available." << std::endl;
		}
	}

	Jaffe::~Jaffe() {
		if (m_cublas_handle) {
			CUBLAS_CHECK(cublasDestroy(m_cublas_handle));
		}
	}

	void Jaffe::SetDevice(const int device_id) {
		int current_device;
		CUDA_CHECK(cudaGetDevice(&current_device));
		if (current_device == device_id) {
			return;
		}

		CUDA_CHECK(cudaSetDevice(device_id));
		if (Get().m_cublas_handle) {
			CUBLAS_CHECK(cublasDestroy(Get().m_cublas_handle));
		}
		CUBLAS_CHECK(cublasCreate(&Get().m_cublas_handle));
	}

	void Jaffe::DeviceQuery() {
		cudaDeviceProp prop;
		int device;
		if (cudaGetDevice(&device) != cudaSuccess) {
			std::cout << "No cuda device present" << endl;
			return;
		}
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
		std::cout << "Device id:					" << device << std::endl;
		std::cout << "Major revision number:		" << prop.major << std::endl;
		std::cout << "Minor revision number:		" << prop.minor << std::endl;
		std::cout << "Name: 						" << prop.name << std::endl;
		std::cout << "Total global memory:			" << prop.totalGlobalMem << 
			std::endl;
		std::cout << "Total shared memory per block:" << prop.sharedMemPerBlock 
			<< std::endl;
		std::cout << "Total registers per block: 	" << prop.regsPerBlock << 
			std::endl;
		std::cout << "Warp size: 					" << prop.warpSize << std::endl;
		std::cout << "Maximum memory pitch:			" << prop.memPitch << std::endl;
		std::cout << "Maximum threads per block:	" << prop.maxThreadsPerBlock
			<< std::endl;
  		std::cout << "Maximum dimension of block:   " << prop.maxThreadsDim[0] << 
			", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << 
			std::endl;
  		std::cout << "Maximum dimension of grid:    " << prop.maxGridSize[0] << ", " 
			<< prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
  		std::cout << "Clock rate:                   " << prop.clockRate << std::endl;
  		std::cout << "Total constant memory:        " << prop.totalConstMem <<
			std::endl;
  		std::cout << "Texture alignment:            " << prop.textureAlignment <<
			std::endl;
  		std::cout << "Concurrent copy and execution:" << (prop.deviceOverlap ? "Yes" 
			: "No") << std::endl;
  		std::cout << "Number of multiprocessors:    " << prop.multiProcessorCount <<
			std::endl;
  		std::cout << "Kernel execution timeout:     " << 
			(prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
  		return;
	}

	bool Jaffe::CheckDevice(const int device_id) {
		bool r = ((cudaSetDevice(device_id)) && (cudaFree(0) == cudaSuccess));
		cudaGetLastError();
		return r;
	}

	int Jaffe::FindDevice(const int start_id) {
		int count = 0;
		CUDA_CHECK(cudaGetDeviceCount(&count));
		for (int i = start_id; i < count; i ++) {
			if (CheckDevice(i)) {
				return i;
			}
		}
		return -1;
	}

	const char* cublasGetErrorString(cublasStatus_t error) {
  		switch (error) {
  		case CUBLAS_STATUS_SUCCESS:
    		return "CUBLAS_STATUS_SUCCESS";
  		case CUBLAS_STATUS_NOT_INITIALIZED:
    		return "CUBLAS_STATUS_NOT_INITIALIZED";
  		case CUBLAS_STATUS_ALLOC_FAILED:
    		return "CUBLAS_STATUS_ALLOC_FAILED";
  		case CUBLAS_STATUS_INVALID_VALUE:
    		return "CUBLAS_STATUS_INVALID_VALUE";
  		case CUBLAS_STATUS_ARCH_MISMATCH:
    		return "CUBLAS_STATUS_ARCH_MISMATCH";
  		case CUBLAS_STATUS_MAPPING_ERROR:
    		return "CUBLAS_STATUS_MAPPING_ERROR";
  		case CUBLAS_STATUS_EXECUTION_FAILED:
    		return "CUBLAS_STATUS_EXECUTION_FAILED";
  		case CUBLAS_STATUS_INTERNAL_ERROR:
    		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  		case CUBLAS_STATUS_NOT_SUPPORTED:
    		return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  		case CUBLAS_STATUS_LICENSE_ERROR:
    		return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  		}	
  		return "Unknown cublas status";
	}
#endif
} // namespace jaffe
