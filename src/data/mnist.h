/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:31:12
 */
#ifndef SRC_MNIST_H_
#define SRC_MNIST_H_

#include <fstream>
#include <iostream>
#include <string>
#include "..\utils.h"

class MNIST {
private:
	std::string data_dir;
	// data_dir 文件的路径

public:
	Matric train_data;
	// 矩阵对象 自己定义的 内部元素是 float 类型 
	// 矩阵的大小是动态的
	Matric train_labels;
	Matric test_data;
	Matric test_labels;

	void read_mnist_data(std::string filename, Matric& data);
	void read_mnist_label(std::string filename, Matric& labels);

	explicit MNIST(std::string data_dir) : data_dir(data_dir) {}
	// explicit 不能隐式转换 建立对象
	void read();
};

#endif  // SRC_MNIST_H_
