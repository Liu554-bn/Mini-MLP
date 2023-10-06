/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-08-24 17:05:03
 */
#include "mnist.h"

int ReverseInt(int i) {
	// 大端存储转小端存储
	// 整型 四个字节
	unsigned char ch1, ch2, ch3, ch4;
	// 无符号字符 1个字节
	ch1 = i & 255;
	// & 与运算 获取最右边的8位
	ch2 = (i >> 8) & 255;
	// i >> 8 i右移8位
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	// (int)ch1 把无符号字符转换成整型 左移24位
	// + ch4 加法运算时会被自动转换为整型
}

void MNIST::read_mnist_data(std::string filename, Matric& data) {
	std::ifstream file(filename, std::ios::binary);
	// 以二进制模式打开 数据文件
	if (file.is_open()) {
		// is_open() 判断文件是否打开
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		// 读取前4个字节的数据 
		// read 只能用字符指针 所以用到强制类型转换
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		// 大端转小端
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
		// n_rows n_cols 28
		// number_of_images 60000
		data.resize(n_cols * n_rows, number_of_images);
		// data 行数是 图片的像素个数 列数是 样本的个数
		// data 784*60000
		for (int i = 0; i < number_of_images; i++) {
			// 每一个样本
			for (int r = 0; r < n_rows; r++) {
				// 样本每行
				for (int c = 0; c < n_cols; c++) {
					// 样本每列
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					// 读一个字节 存到 无符号字符 image 中
					data(r * n_cols + c, i) = (float)image;
					// 图像按行读取 拉长成列存到矩阵里边
				}
			}
		}
	}
}

void MNIST::read_mnist_label(std::string filename, Matric& labels) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		// 把整型 转换成小端存储
		labels.resize(1, number_of_images);
		// 行向量存储标签
		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			// 一次读取一个字节
			labels(0, i) = (float)label;
		}
	}
}

void MNIST::read() {
	read_mnist_data(data_dir + "train-images.idx3-ubyte", train_data);
	read_mnist_data(data_dir + "t10k-images.idx3-ubyte", test_data);
	read_mnist_label(data_dir + "train-labels.idx1-ubyte", train_labels);
	read_mnist_label(data_dir + "t10k-labels.idx1-ubyte", test_labels);
}


