#ifndef SRC_LAYER_SIGMOID_H_
#define SRC_LAYER_SIGMOID_H_

#include "..\layer.h"

class Sigmoid : public Layer {
public:
	void forward(const Matric& bottom);
	void backward(const Matric& bottom, const Matric& grad_top);
};

#endif  // SRC_LAYER_SIGMOID_H_
