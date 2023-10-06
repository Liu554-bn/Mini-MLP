/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 18:59:45
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-08-28 15:37:11
 */
#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 
#include <vector>

#include "utils.h"
#include "optimizer.h"

class Layer {
	// 声明了反向传播等函数

public:
	Matric top;  // 每层的输出
	Matric grad_bottom;  // 向底层传播的梯度
	virtual ~Layer() {}

	virtual void forward(const Matric& bottom) = 0;
	virtual void backward(const Matric& bottom, const Matric& grad_top) = 0;
	virtual void update(Optimizer& opt) {}
	virtual const Matric& output() { return top; }
	virtual const Matric& back_gradient() { return grad_bottom; }
	virtual int output_dim() { return -1; }
	virtual std::vector<float> get_parameters() const
	{
		return std::vector<float>();
	}
	virtual std::vector<float> get_derivatives() const
	{
		return std::vector<float>();
	}
	virtual void set_parameters(const std::vector<float>& param) {}
};

#endif  // SRC_LAYER_H_

