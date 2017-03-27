#ifndef FILLER_H_H
#define FILLER_H_H

#include <string>

#include "blob.h"

namespace jaffe {
	
	template <typename Dtype>
	class JFiller {
	public:
		explicit JFiller() {}
		virtual ~JFiller() {}
		virtual void Fill(JBlob<Dtype>* blob) = 0;

	protected:
		FillerParameter m_filler_param;

	}; // class JFiller

	template <typename Dtype>
	class JConstantFiller : public JFiller<Dtype> {
	public:
		explicit JConstantFiller() : JFiller<Dtype>() {}
		virtual void Fill(JBlob<Dtype>* blob) {
			Dtype* data = blob->GetMutableData();
			const int count = blob->GetCount();
			//======初始化值为0=======
			for (int i = 0; i < count; i++)
				data[i] = 0;
		}
	}; // class JConstantFiller

	template <typename Dtype>
	JFiller<Dtype>* GetFiller() {
		return new JConstantFiller<Dtype>();
	}

} // namespace jaffe
#endif
