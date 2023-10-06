/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:32:00
 */
#ifndef SRC_OPTIMIZER_SGD_H_
#define SRC_OPTIMIZER_SGD_H_

#include <unordered_map>
#include "..\optimizer.h"

class SGD : public Optimizer {
	// 继承自 Optimizer
private:
	std::unordered_map<const float*, Vector> v_map;
	// v_map 每个层更新参数都用得上，所以要搞个映射存储

public:
	explicit SGD(float lr = 0.01) : 
	Optimizer(lr) {}
	// Optimizer(lr, decay) 显式调用基类的构造函数以初始化基类的成员变量
	// momentum(momentum), nesterov(nesterov)
	// 这是成员初始化列表，用于对类的成员变量进行初始化。
	void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};

#endif  // SRC_OPTIMIZER_SGD_H_
