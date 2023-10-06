/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 22:35:39
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-29 10:49:44
 */

#include "rmse_loss.h"

void RMSE::evaluate(const Matric& pred, const Matric& target) {
	int n = pred.cols();
	// n 是样本的个数
	// pred target 10*128
	Matric diff = pred - target;
	// diff 10*128 预测的差值
	loss = diff.cwiseProduct(diff).sum();
	// cwiseProduct 矩阵元素对应元素相乘 
	// sum() 矩阵的元素相加
	loss /= n;
    loss = sqrt(loss);
	// 计算平均损失
	float loss_0 = pow(loss, -0.5);
	grad_bottom = diff * loss_0 / n;
	// grad_bottom 10*128 
	// 损失是标量 对每个样本 列向量y 求梯度 结果还是列向量
}
