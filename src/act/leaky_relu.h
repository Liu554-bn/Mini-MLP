/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 19:22:22
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 19:38:38
 */

#ifndef SRC_LAYER_LEAKY_RELU_H_
#define SRC_LAYER_LEAKY_RELU_H_

#include "..\layer.h"

class Leaky_ReLU : public Layer {
public:
    
	void forward(const Matric& bottom);
	void backward(const Matric& bottom, const Matric& grad_top);
private:
    float alpha = 0.01;
};

#endif  // SRC_LAYER_RELU_H_