/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 16:14:45
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:51:37
 */
#include "..\src\data\mnist.cpp"

int main() {
	MNIST dataset("G:/WorkSpace/VScode/data/");
  	dataset.read();
	int n_train = dataset.train_data.cols();
  	int dim_in = dataset.train_data.rows();
  	std::cout << "mnist train number: " << n_train << std::endl;
  	std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
}