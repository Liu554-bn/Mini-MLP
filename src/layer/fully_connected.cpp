/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-24 16:31:47
 */
#include "fully_connected.h"

void FullyConnected::init() {
	weight.resize(dim_in, dim_out);
	// weight n*p
	bias.resize(dim_out);
	// bias p*1
	grad_weight.resize(dim_in, dim_out);
	// grad_weight n*p
	grad_bias.resize(dim_out);
	// grad_bias p*1
	set_normal_random(weight.data(), weight.size(), 0, 0.01);
	set_normal_random(bias.data(), bias.size(), 0, 0.01);
	// 初始化参数
}

void FullyConnected::forward(const Matric& bottom) {
	// z = w' * x + b
	// bottom 是输入的特征 n*m 
	// top 是输出的特征 p*m
	const int n_sample = bottom.cols();
	// n_sample m 样本的个数
	top.resize(dim_out, n_sample);
	// top p*m
	top = weight.transpose() * bottom;
	// weight.transpose() p*n
	// top p*m
	top.colwise() += bias;
	// colwise() 获取矩阵的每一列
	// 全连接的输出 行是 特征的维度 列是批次的个数
}

void FullyConnected::backward(const Matric& bottom, const Matric& grad_top) {
	// bottom 是输入的特征x n*m
	// grad_top 是损失传来的上游梯度 p*m
	// grad_bottom 损失对x的梯度 n*m
	// grad_weight 损失对w的梯度 n*p
	// grad_bias 损失对b的梯度 p*1
	const int n_sample = bottom.cols();
	// n_sample 样本个数 m
	grad_weight = bottom * grad_top.transpose();
	// grad_weight x*(dl/dz)^T
	grad_bias = grad_top.rowwise().sum();
	// grad_top p*m
	// rowwise().sum() 对每行进行操作 sum() 每行相加
	// grad_bias p*1
	grad_bottom.resize(dim_in, n_sample); 
	grad_bottom = weight * grad_top;
	// grad_bottom w*(dl/dz) n*m
}

void FullyConnected::update(Optimizer& opt) {
	Vector::AlignedMapType weight_vec(weight.data(), weight.size());
	// weight.data() 是权重的指针
	// weight.size() 是矩阵中元素的数量
	// weight_vec 是一个权重矩阵初始的一维数组
	Vector::AlignedMapType bias_vec(bias.data(), bias.size());
	Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(),
		grad_weight.size());
		// ConstAlignedMapType 仅读取数据
	Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

	opt.update(weight_vec, grad_weight_vec);
	opt.update(bias_vec, grad_bias_vec);
	// 更新参数 weight_vec bias_vec
}

std::vector<float> FullyConnected::get_parameters() const {
	std::vector<float> res(weight.size() + bias.size());
	// 新建一个浮点数组 res
	std::copy(weight.data(), weight.data() + weight.size(), res.begin());
	// 将weight指向的元素复制到res指向的数组中
	std::copy(bias.data(), bias.data() + bias.size(),
		res.begin() + weight.size());
	return res;
}

void FullyConnected::set_parameters(const std::vector<float>& param) {
	// 设置参数
	if (static_cast<int>(param.size()) != weight.size() + bias.size())
	// static_cast<int> 强制转换成整型
		throw std::invalid_argument("Parameter size does not match");
		// 抛出参数异常
	std::copy(param.begin(), param.begin() + weight.size(), weight.data());
	std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> FullyConnected::get_derivatives() const {
	// 以浮点数组返回 还是常量返回
	std::vector<float> res(grad_weight.size() + grad_bias.size());
	// 获得当前层参数的梯度 不知道干啥？
	std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(),
		res.begin());
	std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
		res.begin() + grad_weight.size());
	return res;
}

