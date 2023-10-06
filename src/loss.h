#ifndef SRC_LOSS_H_
#define SRC_LOSS_H_

#include "utils.h"

class Loss {
protected:
	float loss;
	// 损失
	Matric grad_bottom;
	// 传到下游梯度

public:
	virtual ~Loss() {}

	virtual void evaluate(const Matric& pred, const Matric& target) = 0;
	// evaluate纯虚函数 只会被子类实现
	virtual float output() { return loss; }
	// 返回损失
	virtual const Matric& back_gradient() { return grad_bottom; }
	// 反向传播
};

#endif  // SRC_LOSS_H_
