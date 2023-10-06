#include "batch_norm.h"

void BatchNorm::init() {
    // 初始化参数的维度和数据
    gamma.resize(dim_in);
    beta.resize(dim_in);
    grad_gamma.resize(dim_in);
    grad_beta.resize(dim_in);
    mean.resize(dim_in);
    var.resize(dim_in);
    set_normal_random(gamma.data(), gamma.size(), 0, 0.01);
    set_normal_random(beta.data(), beta.size(), 0, 0.01);
}

void BatchNorm::forward(const Matric& bottom) {
    // bottom n*m n是特征维度 m 是batch的个数
    Vector mean = bottom.rowwise().mean();
    // mean n*1
    Vector var = (bottom.colwise() - mean).array().square().rowwise().mean();
    // std n*1

    Matric x_norm = (bottom.colwise() - mean);
    // x_norm m*m
    Vector running_var = (var.array() + eps).sqrt();
    // std n*1
    Matric x_norm_1 = x_norm.array().colwise() / running_var.array();
    // x_norm_1 n*m

    Matric out = x_norm_1.array().colwise() * gamma.array();
    top = out.colwise() + beta;
}

void BatchNorm::backward(const Matric& bottom, const Matric& grad_top) {
    // bottom D*N grad_top D*N
    int n = bottom.cols();

    grad_beta = grad_top.rowwise().sum();
    grad_gamma = top.cwiseProduct(grad_top).rowwise().sum();
    
     
    Matric dx_hat = grad_top.array().colwise() * gamma.array();

    //Vector dsigma 

    Vector dsigma_one = (var.array() + eps).pow(-1.5);
    Vector dsigma_two = -0.5 * dx_hat.cwiseProduct(bottom.colwise() - mean).rowwise().sum();
    Vector dsigma = dsigma_two.cwiseProduct(dsigma_one);

    
    //Vector dmu
    Vector dmu_one = 2 * dsigma.cwiseProduct((bottom.colwise() - mean).rowwise().sum()) / n;
    Vector dmu_two = -1 * (dx_hat.array().colwise() / (var.array() + eps).sqrt()).rowwise().sum();
    Vector dmu =  dmu_two - dmu_one;
    
    
    
    //Matric dx
    Matric dx_one = dx_hat.array().colwise() / (var.array() + eps).sqrt();
    Matric dx_two = 2 * ((bottom.colwise() - mean).array().colwise() * dsigma.array()) / n;
    Vector dx_three = dmu.array() / n;
    grad_bottom = (dx_one + dx_two).colwise() + dx_three;
    
}

void BatchNorm::update(Optimizer& opt) {
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

std::vector<float> BatchNorm::get_parameters() const {
	std::vector<float> res(gamma.size() + beta.size());
	// 新建一个浮点数组 res
	std::copy(gamma.data(), gamma.data() + gamma.size(), res.begin());
	// 将weight指向的元素复制到res指向的数组中
	std::copy(beta.data(), beta.data() + beta.size(),
		res.begin() + gamma.size());
	return res;
}

void BatchNorm::set_parameters(const std::vector<float>& param) {
	// 设置参数
	if (static_cast<int>(param.size()) != gamma.size() + beta.size())
	// static_cast<int> 强制转换成整型
		throw std::invalid_argument("Parameter size does not match");
		// 抛出参数异常
	std::copy(param.begin(), param.begin() + gamma.size(), gamma.data());
	std::copy(param.begin() + gamma.size(), param.end(), beta.data());
}

std::vector<float> BatchNorm::get_derivatives() const {
	// 以浮点数组返回 还是常量返回
	std::vector<float> res(grad_gamma.size() + grad_beta.size());
	// 获得当前层参数的梯度 不知道干啥？
	std::copy(grad_gamma.data(), grad_gamma.data() + grad_gamma.size(),
		res.begin());
	std::copy(grad_beta.data(), grad_beta.data() + grad_beta.size(),
		res.begin() + grad_gamma.size());
	return res;
}


