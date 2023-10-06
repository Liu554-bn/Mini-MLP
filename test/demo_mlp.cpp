/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-05-19 14:37:44
 */

#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 
#include <algorithm>
#include <iostream>

#include "..\src\layer.h"
#include "..\src\layer\fully_connected.cpp"
#include "..\src\act\relu.cpp"
#include "..\src\act\sigmoid.cpp"
#include "..\src\act\softmax.cpp"
#include "..\src\loss.h"
#include "..\src\loss\cross_entropy_loss.cpp"
#include "..\src\loss\mse_loss.cpp"
#include "..\src\data\mnist.cpp"
#include "..\src\network.cpp"
#include "..\src\optimizer.h"
#include "..\src\opt\sgd.cpp"
#include "..\src\opt\mome.cpp"
#include "..\src\opt\rmsprop.cpp"
#include "..\src\opt\adam.cpp"
#include "..\src\layer\batch_norm.cpp"
#include "..\src\layer\layer_norm.cpp"
#include <fstream>


bool fileExists(const std::string& filename) {
    std::ifstream ifile(filename.c_str());
    return ifile.good();
}

int main() {

	std::cout << "hello" << std::endl;
	const std::string filename = "../output/train_results.csv";
    if (fileExists(filename)) {
        std::remove(filename.c_str());
    } 
	std::ofstream ofs("../output/train_results.csv");
	ofs << "Epoch,Loss,Accuracy\n";
	MNIST dataset("G:/WorkSpace/VScode/MLP/mini_MLP/dataSet/");
	dataset.read();
	int n_train = dataset.train_data.cols();
	// n_train 训练集的个数
	int dim_in = dataset.train_data.rows();
	// dim_in 一张图像的像素数 特征的维度
	std::cout << "mnist train number: " << n_train << std::endl;
	std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
	// 60000 10000
	// dnn
	Network dnn;
    Layer* fc1 = new FullyConnected(784, 256);
    Layer* fc2 = new FullyConnected(256, 128);
	Layer* fc3 = new FullyConnected(128, 64);
	Layer* fc4 = new FullyConnected(64, 10);
	Layer* relu1 = new ReLU;
	Layer* relu2 = new ReLU;
	Layer* relu3 = new ReLU;
	Layer* batch = new BatchNorm(64);
	Layer* layer = new LayerNorm(64);
	Layer* softmax = new Softmax;

    dnn.add_layer(fc1);
    dnn.add_layer(relu1);
	dnn.add_layer(fc2);
    dnn.add_layer(relu2);
	dnn.add_layer(fc3);
	dnn.add_layer(relu3);
	dnn.add_layer(layer);
	dnn.add_layer(fc4);
	dnn.add_layer(softmax);
	
	Loss* loss = new MSE;
	dnn.add_loss(loss);
	// train & test
	// MOME opt(0.001, 0.9);
	// SGD opt(0.001);
	// RMSProp opt(0.001, 0.99, 1e-8);
	Adam opt(0.001, 0.9,0.999, 1e-8);
	const int n_epoch = 1;
	const int batch_size = 128;
	for (int epoch = 0; epoch < n_epoch; epoch++) {
		shuffle_data(dataset.train_data, dataset.train_labels);
		float train_loss = 0;
		int ith_batch = 0;
		for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
			// n_train 60000
			ith_batch = start_idx / batch_size;
			// ith_batch 0 记录第几个batch 从0开始
			Matric x_batch = dataset.train_data.block(0, start_idx, dim_in,
				std::min(batch_size, n_train - start_idx));
				// dim_in 784
				// block 提取矩阵 
				// std::min(batch_size, n_train - start_idx) 是提取矩阵的列数
				// 有可能最后一个矩阵到不了 batch_size 的大小
				// 所以选一个最小的作为 矩阵的列数
				// x_batch 784*128
			Matric label_batch = dataset.train_labels.block(0, start_idx, 1,
				std::min(batch_size, n_train - start_idx));
				// label_batch 1*128
			Matric target_batch = one_hot_encode(label_batch, 10);
			// target_batch 10*128
			dnn.forward(x_batch);// 前向传播
			dnn.backward(x_batch, target_batch);// 反向传播
			// display
			train_loss += dnn.get_loss();
			// optimize
			dnn.update(opt);
		}
		train_loss = train_loss / (ith_batch + 1);
		// test
		dnn.forward(dataset.test_data);
		float acc = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << dataset.test_labels.cols() << std::endl;
		std::cout << dataset.test_labels.rows() << std::endl;
		std::cout << dnn.output().cols() << std::endl;
		std::cout << dnn.output().rows() << std::endl;
		// 
		std::cout << std::endl;
		std::cout << epoch << " epoch test acc: " << acc << std::endl;
		ofs << epoch << "," << dnn.get_loss() << "," << acc << "\n";
	}

	ofs.close();

	return 0;
}

