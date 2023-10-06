#ifndef SRC_LAYER_SOFTMAX_H_
#define SRC_LAYER_SOFTMAX_H_

#include "..\layer.h"

class Softmax : public Layer {
public:
	void forward(const Matric& bottom);
	// bottom 输入
	void backward(const Matric& bottom, const Matric& grad_top);
	// grad_top 上游梯度
};

#endif  // SRC_LAYER_SOFTMAX_H_
