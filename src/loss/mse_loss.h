#ifndef SRC_LOSS_MSE_LOSS_H_
#define SRC_LOSS_MSE_LOSS_H_

#include "..\loss.h"

class MSE : public Loss {
public:
	void evaluate(const Matric& pred, const Matric& target);
	// 计算损失 和 下游梯度
};

#endif  // SRC_LOSS_MSE_LOSS_H_
