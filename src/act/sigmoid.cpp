#include "sigmoid.h"

void Sigmoid::forward(const Matric& bottom) {
	// a = 1 / (1 + exp(-z))
	top.array() = 1.0 / (1.0 + (-bottom).array().exp());
}

void Sigmoid::backward(const Matric& bottom, const Matric& grad_top) {
	// d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
	// d(a_i)/d(z_i) = a_i * (1-a_i)
	Matric da_dz = top.array().cwiseProduct(1.0 - top.array());
	grad_bottom = grad_top.cwiseProduct(da_dz);
	// grad_bottom 下游梯度
}
