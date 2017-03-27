// huangshize 2016.03.28
// === rawdata.h ===
// SyncedMemory.hpp的替代，用来存储原始数据
// 没有实现cpu数据与gpu数据的同步

#ifndef JAFFE_RAWDATA_H_
#define JAFFE_RAWDATA_H_

#include <cstdlib>
#include "common.h"

namespace jaffe {
    // 为指针开辟空间
    inline void JaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
	//	cout << __FILE__ << "\t" << __LINE__ << "\tJaffeMallocHost()" << endl;
		if (Jaffe::GetMode() == Jaffe::GPU) {
			CUDA_CHECK(cudaMallocHost(ptr, size));
			*use_cuda = true;
			return;
		}
#endif
        *ptr = malloc(size);
		*use_cuda = false;
    }
	
    // 回收空间
    inline void JaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
		if (use_cuda) {
			CUDA_CHECK(cudaFreeHost(ptr));
			return;
		}
#endif
        free(ptr);
    }

    class JRawData {
    public:
        JRawData() 
			: m_cpu_ptr(NULL), m_gpu_ptr(NULL), m_size(0), m_head(UNINITIALIZED),
			m_own_cpu_data(false), m_cpu_malloc_use_cuda(false), 
			m_own_gpu_data(false), m_gpu_device(-1) {}
        explicit JRawData(size_t size) 
			: m_cpu_ptr(NULL), m_gpu_ptr(NULL), m_size(size), m_head(UNINITIALIZED), 
			m_own_cpu_data(false), m_cpu_malloc_use_cuda(false), 
		    m_own_gpu_data(false), m_gpu_device(-1)	{}
        ~JRawData();

        // 返回RawData中数据的大小
        size_t GetSize() { return m_size; }
        //
        bool GetOwnCpuData() { return m_own_cpu_data; }
        // 获取data的指针
        const void* GetCpuData();
		const void* GetGpuData();
        void* GetMutableCpuData();
		void* GetMutableGpuData();
		void SetCpuData(void* data);
		void SetGpuData(void* data);
		inline void GetTag() {
			cout << m_head << endl;
		}

		enum RawHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };

    private:
        void ToCpu();  // to_cpu() 
		void ToGpu();  // to_gpu()
        void* m_cpu_ptr;
		void* m_gpu_ptr;
        size_t m_size;
        bool m_own_cpu_data;
		bool m_own_gpu_data;
		RawHead m_head;
		bool m_cpu_malloc_use_cuda;
		int m_gpu_device;
    };
}
#endif
