/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 20:24:19
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:30:24
 */

#ifndef SRC_LAYER_LEAKY_RELU_H_
#define SRC_LAYER_LEAKY_RELU_H_

#include "..\layer.h"

class Leaky_ReLU : public Layer {
public:
    float alpha = 0.01;
	void forward(const Matric& bottom);
	void backward(const Matric& bottom, const Matric& grad_top);
};

#endif  // SRC_LAYER_RELU_H_