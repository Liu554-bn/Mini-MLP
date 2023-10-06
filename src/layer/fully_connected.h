/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-20 20:47:22
 */
#ifndef SRC_LAYER_FULLY_CONNECTED_H_
#define SRC_LAYER_FULLY_CONNECTED_H_

#include <vector>
#include "..\layer.h"

class FullyConnected : public Layer {
	// 继承自类 Layer
private:
	const int dim_in;
	// 输入特征 n
	const int dim_out;
	// 输出特征 p

	Matric weight;  // 权重w n*p
	Vector bias;  // 偏置b p*1
	Matric grad_weight;  // 权重的梯度
	Vector grad_bias;  // 偏置的梯度

	void init();

public:
	FullyConnected(const int dim_in, const int dim_out) :
		dim_in(dim_in), dim_out(dim_out)
		// 构造函数
	{
		init();
	}

	void forward(const Matric& bottom);
	void backward(const Matric& bottom, const Matric& grad_top);
	void update(Optimizer& opt);
	int output_dim() { return dim_out; }
	std::vector<float> get_parameters() const;
	std::vector<float> get_derivatives() const;
	void set_parameters(const std::vector<float>& param);
};

#endif  // SRC_LAYER_FULLY_CONNECTED_H_
