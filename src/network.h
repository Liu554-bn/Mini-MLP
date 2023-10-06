/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-20 20:41:19
 */
#ifndef SRC_NETWORK_H_
#define SRC_NETWORK_H_

#include <stdlib.h>
#include <vector>
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "utils.h"

class Network {
private:
	std::vector<Layer*> layers;  
	//  layers Layer指针组成的动态数组
	Loss* loss;
	// loss 损失指针

public:
	Network() : loss(NULL) {}
	~Network() {
		for (int i = 0; i < layers.size(); i++) {
			delete layers[i];
			// 删除 layers数组中的指针
		}
		if (loss) {
			delete loss;
			// 删除数组指针
		}
	}

	void add_layer(Layer* layer) { layers.push_back(layer); }
	// 把层指针加到数组中
	void add_loss(Loss* loss_in) { loss = loss_in; }
	// 设定损失指针

	void forward(const Matric& input);
	void backward(const Matric& input, const Matric& target);
	void update(Optimizer& opt);
	// update 更新参数

	const Matric& output() { return layers.back()->output(); }
	// const Matrix& 常量引用返回值 不会对返回值进行修改
	// 返回前向传播的输出
	float get_loss() { return loss->output(); }
	// 获取损失
	std::vector<std::vector<float>> get_parameters() const;
	// const 说明 get_parameters()是一个常函数 不会修改类内的变量
	void set_parameters(const std::vector< std::vector<float> >& param);
	// 设定参数
	std::vector<std::vector<float> > get_derivatives() const;
	// 
	void check_gradient(const Matric& input, const Matric& target, int n_points,
		int seed = -1);
	// 检查梯度
};

#endif  // SRC_NETWORK_H_
