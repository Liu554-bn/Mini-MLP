#include "network.h"

void Network::forward(const Matric& input) {
	if (layers.empty())
		return;
	layers[0]->forward(input);
	for (int i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->output());
	}
}

void Network::backward(const Matric& input, const Matric& target) {
	int n_layer = layers.size();
	// n_layer 层的个数
	if (n_layer <= 0)
		return;
 
	loss->evaluate(layers[n_layer - 1]->output(), target);
	// 计算平均损失
	if (n_layer == 1) {
		// 只有一层
		layers[0]->backward(input, loss->back_gradient());
		// loss->back_gradient() 损失函数的上层梯度
		// layers[0]->backward 计算这一层下游梯度和这层参数的梯度
		return;
	}
	// >1 layers
	layers[n_layer - 1]->backward(layers[n_layer - 2]->output(),
		loss->back_gradient());
	for (int i = n_layer - 2; i > 0; i--) {
		layers[i]->backward(layers[i - 1]->output(), layers[i + 1]->back_gradient());
	}
	layers[0]->backward(input, layers[1]->back_gradient());
	// 计算每一层的梯度，和对应的下游梯度
}

void Network::update(Optimizer& opt) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->update(opt);
		// 用相应的优化器更新参数
	}
}

std::vector<std::vector<float> > Network::get_parameters() const {
	const int n_layer = layers.size();
	// n_layer 层的个数
	std::vector< std::vector<float> > res;
	res.reserve(n_layer);
	// 先分配空间
	for (int i = 0; i < n_layer; i++) {
		res.push_back(layers[i]->get_parameters());
	}
	return res;
}

void Network::set_parameters(const std::vector< std::vector<float> >& param) {
	const int n_layer = layers.size();
	if (static_cast<int>(param.size()) != n_layer)
	// param参数的层数和当前的层数不匹配
		throw std::invalid_argument("Parameter size does not match");
	for (int i = 0; i < n_layer; i++) {
		// 设置参数
		layers[i]->set_parameters(param[i]);
	}
}

std::vector<std::vector<float> > Network::get_derivatives() const {
	// 获取每一层的参数的梯度
	const int n_layer = layers.size();
	std::vector< std::vector<float> > res;
	res.reserve(n_layer);
	// 提前分配内存
	for (int i = 0; i < n_layer; i++) {
		res.push_back(layers[i]->get_derivatives());
		// 把每一层参数的梯度组成的动态数组存入动态数组
	}
	return res;
}

void Network::check_gradient(const Matric& input, const Matric& target,
	int n_points, int seed) {
	if (seed > 0)
		std::srand(seed);

	this->forward(input);
	this->backward(input, target);
	std::vector< std::vector<float> > param = this->get_parameters();
	std::vector< std::vector<float> > deriv = this->get_derivatives();

	const float eps = 1e-4;
	const int n_layer = deriv.size();
	for (int i = 0; i < n_points; i++) {
		// Randomly select a layer
		const int layer_id = int(std::rand() / double(RAND_MAX) * n_layer);
		// Randomly pick a parameter, note that some layers may have no parameters
		const int n_param = deriv[layer_id].size();
		if (n_param < 1)  continue;
		const int param_id = int(std::rand() / double(RAND_MAX) * n_param);
		// Turbulate the parameter a little bit
		const float old = param[layer_id][param_id];

		param[layer_id][param_id] -= eps;
		this->set_parameters(param);
		this->forward(input);
		this->backward(input, target);
		const float loss_pre = loss->output();

		param[layer_id][param_id] += eps * 2;
		this->set_parameters(param);
		this->forward(input);
		this->backward(input, target);
		const float loss_post = loss->output();

		const float deriv_est = (loss_post - loss_pre) / eps / 2;

		std::cout << "[layer " << layer_id << ", param " << param_id <<
			"] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est <<
			", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;

		param[layer_id][param_id] = old;
	}

	// Restore original parameters
	this->set_parameters(param);
}
