/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 18:59:45
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 19:03:53
 */
#ifndef SRC_LAYER_RELU_H_
#define SRC_LAYER_RELU_H_

#include "..\layer.h"

class ReLU : public Layer {
public:
	void forward(const Matric& bottom);
	void backward(const Matric& bottom, const Matric& grad_top);
};

#endif  // SRC_LAYER_RELU_H_
