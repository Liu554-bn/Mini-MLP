#ifndef SRC_LOSS_CROSS_ENTROPY_LOSS_H_
#define SRC_LOSS_CROSS_ENTROPY_LOSS_H_

#include "..\loss.h"

class CrossEntropy : public Loss {
public:
	void evaluate(const Matric& pred, const Matric& target);
};// 计算交叉熵损失

#endif  // SRC_LOSS_CROSS_ENTROPY_LOSS_H_
