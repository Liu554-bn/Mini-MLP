/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 19:22:12
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 19:35:28
 */
#include "leaky_relu.h"

void Leaky_ReLU::forward(const Matric& bottom) {
	
    
    // alpha是系数
	top = bottom.array().max(alpha * bottom.array());
	// 
}

void Leaky_ReLU::backward(const Matric& bottom, const Matric& grad_top) {
	
	Matric positive = (bottom.array() > 0.0).cast<float>();
	Matric negative = (bottom.array() <= 0.0).cast<float>() * alpha;
	grad_bottom = grad_top.cwiseProduct(positive) + grad_top.cwiseProduct(negative);
}
