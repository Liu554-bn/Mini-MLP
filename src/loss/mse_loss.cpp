/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-25 20:26:46
 */
#include "mse_loss.h"

void MSE::evaluate(const Matric& pred, const Matric& target) {
	int n = pred.cols();
	// n 是样本的个数
	// pred target 10*128
	Matric diff = pred - target;
	// diff 10*128 预测的差值
	loss = diff.cwiseProduct(diff).sum();
	// cwiseProduct 矩阵元素对应元素相乘 
	// sum() 矩阵的元素相加
	loss /= n;
	// 计算平均损失
	grad_bottom = diff * 2 / n;
	// grad_bottom 10*128 
	// 损失是标量 对每个样本 列向量y 求梯度 结果还是列向量
}
