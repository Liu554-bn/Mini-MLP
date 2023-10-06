#include "relu.h"

void ReLU::forward(const Matric& bottom) {
	// a = z*(z>0)
	top = bottom.cwiseMax(0.0);
	// cwiseMax(0.0) 将矩阵的元素和0比大小，最终输出最大的值
}

void ReLU::backward(const Matric& bottom, const Matric& grad_top) {
	// d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
	//             = d(L)/d(a_i) * 1*(z_i>0)
	Matric positive = (bottom.array() > 0.0).cast<float>();
	// cast 将矩阵的类型 从 bool 转换成 整型
	grad_bottom = grad_top.cwiseProduct(positive);
}
