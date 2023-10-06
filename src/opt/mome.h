/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 22:02:37
 */
#ifndef SRC_OPTIMIZER_MOME_H_
#define SRC_OPTIMIZER_MOME_H_

#include <unordered_map>
#include "..\optimizer.h"

class MOME : public Optimizer {
	// 继承自 Optimizer
private:
	float momentum;  // 动量 (default: 0)
	std::unordered_map<const float*, Vector> v_map;
	// v_map 每个层更新参数都用得上，所以要搞个映射存储

public:
	explicit MOME(float lr = 0.01, float momentum = 0.9) 
        : Optimizer(lr),momentum(momentum) {}
	// Optimizer(lr, decay) 显式调用基类的构造函数以初始化基类的成员变量
	// momentum(momentum), nesterov(nesterov)
	// 这是成员初始化列表，用于对类的成员变量进行初始化。
	void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};

#endif  // SRC_OPTIMIZER_SGD_H_
