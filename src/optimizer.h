/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-20 19:47:35
 */
#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "utils.h"

class Optimizer {
protected:
	float lr;  // 学习率
	

public:
	explicit Optimizer(float lr = 0.01) :
	// 方式隐式调用
		lr(lr) {}
	virtual ~Optimizer() {}

	virtual void update(Vector::AlignedMapType& w,
	// 虚函数
		Vector::ConstAlignedMapType& dw) = 0;
};

#endif  // SRC_OPTIMIZER_H_
