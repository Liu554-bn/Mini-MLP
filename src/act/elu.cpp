#include "elu.h"

void ELU::forward(const Matric& bottom) {
        // 计算 ELU 函数
        top = bottom.array().max(0) + alpha * (bottom.array().min(0).exp() - 1);
}

void ELU::backward(const Matric& bottom, const Matric& grad_top) {
    Matric positive = (bottom.array() >= 0.0).cast<float>();
	Matric negative = (bottom.array() < 0.0).cast<float>().exp() * alpha;
	grad_bottom = grad_top.cwiseProduct(positive) + grad_top.cwiseProduct(negative);
    
}