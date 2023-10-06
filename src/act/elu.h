#ifndef SRC_LAYER_ELU_H_
#define SRC_LAYER_ELU_H_

#include "..\layer.h"

class ELU : public Layer {
public:
	void forward(const Matric& bottom);
	// bottom 输入
	void backward(const Matric& bottom, const Matric& grad_top);
	// grad_top 上游梯度
private:
    float alpha = 1;
};


#endif  // SRC_LAYER_SOFTMAX_H_
