// huangshize 2016.03.29
// === common.h ===

#ifndef JAFFE_COMMON_H_
#define JAFFE_COMMON_H_

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <utility>  // pair

#include <climits>
#include <cmath>
#include <ctime>
#include <thread>

#include "util/device_alternate.hpp"
#include "proto/jaffe.pb.h"
#include "cmake/jaffe_config.h"
#include "util/math_functions.hpp"

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
	template void classname<float>::ForwardGpu( \
		const std::vector<JBlob<float>*>& bottom, \
		const std::vector<JBlob<float>*>& top); \
	template void classname<double>::ForwardGpu( \
		const std::vector<JBlob<double>*>& bottom, \
		const std::vector<JBlob<double>*>& top);

namespace cv { class Mat; }

namespace jaffe {

	using std::fstream;
	using std::ios;
	using std::isnan;
	using std::isinf;
	using std::iterator;
	using std::make_pair;
	using std::map;
	using std::ostringstream;
	using std::pair;
	using std::set;
	using std::string;
	using std::stringstream;
	using std::vector;
	using std::shared_ptr;
	using std::cout;
	using std::endl;


	// 全局初始化方程
	// void GlobalInit(int* pargc, char*** pargv);

	class Jaffe
	{
	public:
		~Jaffe();

		static Jaffe& Get();

		enum Brew { CPU, GPU };

#ifndef CPU_ONLY
		inline static cublasHandle_t GetCublasHandle() { return Get().m_cublas_handle; }
#endif

		inline static Brew GetMode() { return Get().m_mode; }
		inline static void SetMode(Brew mode) { Get().m_mode = mode; }
		static void SetDevice(const int device_id);
		static void DeviceQuery();
		static bool CheckDevice(const int device_id);
		static int FindDevice(const int start_id = 0);

	protected:
		Brew m_mode;

#ifndef CPU_ONLY
		cublasHandle_t m_cublas_handle;
#endif

	private:
		// 私有化构造函数避免复制实例化
		Jaffe();
	DISABLE_COPY_AND_ASSIGN(Jaffe);

	}; // class Jaffe

} // namespace jaffe
#endif
