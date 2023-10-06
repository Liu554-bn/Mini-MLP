/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 21:16:56
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 12:16:47
 */
#include "layer_norm.h"

void LayerNorm::init() {
    // 初始化参数的维度和数据
    gamma.resize(dim_in);
    beta.resize(dim_in);
    grad_gamma.resize(dim_in);
    grad_beta.resize(dim_in);
    
    set_normal_random(gamma.data(), gamma.size(), 0, 0.01);
    set_normal_random(beta.data(), beta.size(), 0, 0.01);
}

void LayerNorm::forward(const Matric& bottom) {
    // bottom n*m n是特征维度 m 是batch的个数
	int num = bottom.cols();
	mean.resize(num);
    var.resize(num);

    mean = bottom.colwise().mean();
    
    var =  (bottom.rowwise() - mean.transpose()).array().square().colwise().mean();
    
    // Matric
    Matric x_halt = (bottom.rowwise() - mean.transpose()).array().rowwise() / (var.array() + eps).transpose();
    
    top = (x_halt.array().colwise() * gamma.array()).colwise() + beta.array();
    
}

void LayerNorm::backward(const Matric& bottom, const Matric& grad_top) {
    // bottom D*N grad_top D*N
    int n = dim_in;

	grad_gamma = grad_top.rowwise().sum();
    grad_beta = top.cwiseProduct(grad_top).rowwise().sum();

    Matric dx_hat = grad_top.array().colwise() * gamma.array();
          
    Vector dsigma_one = -0.5 * (bottom.rowwise() - mean.transpose()).colwise().sum();
    Vector dsigma_two = (var.array() + eps).pow(-1.5);
    Vector dsigma = dsigma_one.cwiseProduct(dsigma_two);
    
    Vector dmu_one = -1 * (dx_hat.array().rowwise() / (var.array() + eps).sqrt().transpose()).colwise().sum();
    Vector dmu_two = 2 * dsigma.cwiseProduct((bottom.rowwise() - mean.transpose()).colwise().sum().transpose()) / n;
    Vector dmu =  dmu_one - dmu_two;
    

    Matric dx_one = dx_hat.array().rowwise() / (var.array() + eps).sqrt().transpose();  
    Matric dx_two = 2 * ((bottom.rowwise() - mean.transpose()).array().rowwise() * dsigma.array().transpose()) / n;
    Vector dx_three = dmu.array() / n;
    grad_bottom = (dx_one + dx_two).rowwise() + dx_three.transpose();
    
}

void LayerNorm::update(Optimizer& opt) {
	Vector::AlignedMapType gamma_vec(gamma.data(), gamma.size());
	// weight.data() 是权重的指针
	// weight.size() 是矩阵中元素的数量
	// weight_vec 是一个权重矩阵初始的一维数组
	Vector::AlignedMapType beta_vec(beta.data(), beta.size());
	Vector::ConstAlignedMapType grad_gamma_vec(grad_gamma.data(),
		grad_gamma.size());
		// ConstAlignedMapType 仅读取数据
	Vector::ConstAlignedMapType grad_beta_vec(grad_beta.data(), grad_beta.size());

	opt.update(gamma_vec, grad_gamma_vec);
	opt.update(beta_vec, grad_beta_vec);
	// 更新参数 weight_vec bias_vec
}

std::vector<float> LayerNorm::get_parameters() const {
	std::vector<float> res(gamma.size() + beta.size());
	// 新建一个浮点数组 res
	std::copy(gamma.data(), gamma.data() + gamma.size(), res.begin());
	// 将weight指向的元素复制到res指向的数组中
	std::copy(beta.data(), beta.data() + beta.size(),
		res.begin() + gamma.size());
	return res;
}

void LayerNorm::set_parameters(const std::vector<float>& param) {
	// 设置参数
	if (static_cast<int>(param.size()) != gamma.size() + beta.size())
	// static_cast<int> 强制转换成整型
		throw std::invalid_argument("Parameter size does not match");
		// 抛出参数异常
	std::copy(param.begin(), param.begin() + gamma.size(), gamma.data());
	std::copy(param.begin() + gamma.size(), param.end(), beta.data());
}

std::vector<float> LayerNorm::get_derivatives() const {
	// 以浮点数组返回 还是常量返回
	std::vector<float> res(grad_gamma.size() + grad_beta.size());
	// 获得当前层参数的梯度 不知道干啥？
	std::copy(grad_gamma.data(), grad_gamma.data() + grad_gamma.size(),
		res.begin());
	std::copy(grad_beta.data(), grad_beta.data() + grad_beta.size(),
		res.begin() + grad_gamma.size());
	return res;
}


