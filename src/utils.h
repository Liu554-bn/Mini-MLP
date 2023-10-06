/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-05-19 15:25:42
 */
#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core"  
#include <algorithm>
#include <iostream>
#include <random>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matric;
// 定义动态矩阵 Matrix
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
// 定义列向量 Vector
typedef Eigen::Array<float, 1, Eigen::Dynamic> RowVector;
// 定义行向量 RowVector
static std::default_random_engine generator;
// generator 随机数生成器
// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(float* arr, int n, float mu, float sigma) {
	// inline 减少函数调用的开销
	// float* arr 浮点数数组 arr
	std::normal_distribution<float> distribution(mu, sigma);
	// distribution 正态分布
	for (int i = 0; i < n; i++) {
		arr[i] = distribution(generator);
		// 存储浮点数
	}
}

// shuffle cols of matrix
inline void shuffle_data(Matric& data, Matric& labels) {
	// data 784*60000 labels 1*60000
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.cols());
	perm.setIdentity();
	// perm是一个置换矩阵 0 和 1 组成
	std::random_shuffle(perm.indices().data(), perm.indices().data()
		+ perm.indices().size());
		// indices() 是矩阵列的下标的排列
	data = data * perm;  // permute columns
	// 左行右列 这里右乘打乱的单位矩阵 做到打乱数据集和标签的目的
	labels = labels * perm;
}

// encode discrete values to one-hot values
inline Matric one_hot_encode(const Matric& y, int n_value) {
	// 将标签转换成独热编码
	// y 1*128 n_value 10
	int n = y.cols();
	// n 128 标签的个数
	Matric y_onehot = Matric::Zero(n_value, n);
	// y_onehot 10*128 全为0的矩阵
	for (int i = 0; i < n; i++) {
		y_onehot(int(y(i)), i) = 1;
		// 对矩阵进行赋值
		// y(i) 样本的标签 0-9 作为行
		// i 样本的下标 作为列
	}
	return y_onehot;
}

// classification accuracy
inline float compute_accuracy(const Matric& preditions, const Matric& labels) {
	// preditions  1*10000 
	// labels 10*10000
	int n = preditions.cols();
	// n 是样本个数
	float acc = 0;
	for (int i = 0; i < n; i++) {
		// i表示样本的个数
		Matric::Index max_index;
		float max_value = preditions.col(i).maxCoeff(&max_index);
		// max_value 每列的最大值
		// max_index 最大值的行索引
		acc += int(max_index) == labels(i);
	}
	return acc / n;
}

inline float compute_precision_recall(const Matric& predictions, const Matric& labels, float& precision, float& recall) {
    
	int num_classes = predictions.rows();
    int num_samples = predictions.cols();

    Matric true_positives = Matric::Zero(num_classes, 1);
    Matric false_positives = Matric::Zero(num_classes, 1);
    Matric false_negatives = Matric::Zero(num_classes, 1);

    for (int i = 0; i < num_samples; i++) {
        int true_class = labels(i);
        int predicted_class;
        predictions.col(i).maxCoeff(&predicted_class);

        if (predicted_class == true_class) {
            true_positives(true_class)++;
        } else {
            false_positives(predicted_class)++;
            false_negatives(true_class)++;
        }
    }

    float total_precision = 0.0;
    float total_recall = 0.0;

    for (int i = 0; i < num_classes; i++) {
        if (true_positives(i) == 0) {
            precision = recall = 0;
            std::cerr << "Warning: true positives for class " << i << " is zero" << std::endl;
        } else {
            precision = true_positives(i) / (true_positives(i) + false_positives(i));
            recall = true_positives(i) / (true_positives(i) + false_negatives(i));
        }
        total_precision += precision;
        total_recall += recall;
    }

    precision = total_precision / num_classes;
    recall = total_recall / num_classes;

    return (2 * precision * recall) / (precision + recall);
}

#endif  // SRC_UTILS_H_
