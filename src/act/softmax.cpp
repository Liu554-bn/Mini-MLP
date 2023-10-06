/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 20:29:56
 */
#include "softmax.h"

void Softmax::forward(const Matric& bottom) {
	// int rows = bottom.rows();
	// int cols = bottom.cols();
	// std::cout << "The shape of pred is " << rows << " x " << cols << std::endl;
	// bottom 10*128
	top.array() = (bottom.rowwise() - bottom.colwise().maxCoeff()).array().exp();
	// bottom.colwise().maxCoeff() 获取矩阵每列的最大值
	// bottom.rowwise() - bottom.colwise().maxCoeff() 每列减去这列最大的值
	// top softmax的分子
	// top 10*128
	RowVector z_exp_sum = top.colwise().sum();  
	// z_exp_sum 行向量 1*128
	top.array().rowwise() /= z_exp_sum;
	// top 10*128
	
}

void Softmax::backward(const Matric& bottom, const Matric& grad_top) {
	// grad_top 
	RowVector temp_sum = top.cwiseProduct(grad_top).colwise().sum();
	// temp_sum 1*128
	grad_bottom.array() = top.array().cwiseProduct(grad_top.array().rowwise()
		- temp_sum);
		// grad_bottom 10*128
}
