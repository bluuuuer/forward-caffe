// huangshize 2016.03.16
// === rawdata.cpp ===
#include "rawdata.h"

namespace jaffe {
    JRawData::~JRawData() {
        if (m_cpu_ptr && m_own_cpu_data)
            JaffeFreeHost(m_cpu_ptr, m_cpu_malloc_use_cuda);
#ifndef CPU_ONLY
		if (m_gpu_ptr && m_own_gpu_data) {
			int initial_device;
			cudaGetDevice(&initial_device);
			if (m_gpu_device != -1) {
				CUDA_CHECK(cudaSetDevice(m_gpu_device));
			}
			CUDA_CHECK(cudaFree(m_gpu_ptr));
			cudaSetDevice(initial_device);
		}
#endif
    }

    void JRawData::ToCpu() {
		switch (m_head) {
		case UNINITIALIZED:
			JaffeMallocHost(&m_cpu_ptr, m_size, &m_cpu_malloc_use_cuda);
			JaffeMemset(m_size, 0, m_cpu_ptr);
			m_head = HEAD_AT_CPU;
			m_own_cpu_data = true;
			break;
		case HEAD_AT_GPU:
#ifndef CPU_ONLY
			if (m_cpu_ptr == NULL) {
				JaffeMallocHost(&m_cpu_ptr, m_size, &m_cpu_malloc_use_cuda);
				m_own_cpu_data = true;
			}
			JaffeGpu2CpuMemcpy(m_size, m_gpu_ptr, m_cpu_ptr);
			m_head = SYNCED;
#else
			NO_GPU;
#endif
			break;
		case HEAD_AT_CPU:
		case SYNCED:
			break;
		}
	}

	void JRawData::ToGpu() {
#ifndef CPU_ONLY
		switch (m_head) {
		case UNINITIALIZED:	
			CUDA_CHECK(cudaGetDevice(&m_gpu_device));
			CUDA_CHECK(cudaMalloc(&m_gpu_ptr, m_size));
			JaffeGpuMemset(m_size, 0, m_gpu_ptr);
			m_head = HEAD_AT_GPU;
			m_own_gpu_data = true;
			break;
		case HEAD_AT_CPU:
			if (m_gpu_ptr == NULL) {
				CUDA_CHECK(cudaGetDevice(&m_gpu_device));
				CUDA_CHECK(cudaMalloc(&m_gpu_ptr, m_size));
				m_own_gpu_data = true;
			}
			JaffeGpuMemcpy(m_size, m_cpu_ptr, m_gpu_ptr);
			m_head = SYNCED;
			break;
		case HEAD_AT_GPU:
		case SYNCED:
			break;

		}
#else
		NO_GPU;
#endif
	}

    const void* JRawData::GetCpuData() {
        ToCpu();
        return (const void*)m_cpu_ptr;
    }
	
	const void* JRawData::GetGpuData() {
#ifndef CPU_ONLY
		ToGpu();
		return (const void*)m_gpu_ptr;
#else 
		NO_GPU;
		return NULL;
#endif
	}

    void* JRawData::GetMutableCpuData() {
        ToCpu();
		m_head = HEAD_AT_CPU;
        return m_cpu_ptr;
    }

	void* JRawData::GetMutableGpuData() {
#ifndef CPU_ONLY
		ToGpu();
		m_head = HEAD_AT_GPU;
		return m_gpu_ptr;
#else
		NO_GPU;
		return NULL;
#endif
	}

    void JRawData::SetCpuData(void* data) {
        if (m_own_cpu_data)
            JaffeFreeHost(m_cpu_ptr, m_cpu_malloc_use_cuda);
        m_cpu_ptr = data;
		m_head = HEAD_AT_CPU;
        m_own_cpu_data = false;
    }

	void JRawData::SetGpuData(void* data) {
#ifndef CPU_ONLY
		if (m_own_gpu_data) {
			int initial_device;
			cudaGetDevice(&initial_device);
			if (m_gpu_device != -1) {
				CUDA_CHECK(cudaSetDevice(m_gpu_device));
			}
			CUDA_CHECK(cudaFree(m_gpu_ptr));
			cudaSetDevice(initial_device);
		}
		m_gpu_ptr = data;
		m_head = HEAD_AT_GPU;
		m_own_gpu_data = false;
#else
		NO_GPU;
#endif
	}
} // namespace jaffe
