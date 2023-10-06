/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-25 20:59:32
 */
#include "cross_entropy_loss.h"

void CrossEntropy::evaluate(const Matric& pred, const Matric& target) {
	int n = pred.cols();
	const float eps = 1e-8;
	// eps 防止计算时出错
	// pred 10*128
	loss = -(target.array().cwiseProduct((pred.array() + eps).log())).sum();
	// 
	loss /= n;
	// loss是平均损失
	grad_bottom = -target.array().cwiseQuotient(pred.array() + eps) / n;
	// grad_bottom 10*128
}
