/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 22:35:46
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:31:38
 */
/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 22:35:46
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:31:34
 */
#ifndef SRC_LOSS_RMSE_LOSS_H_
#define SRC_LOSS_RMSE_LOSS_H_

#include "..\loss.h"

class RMSE : public Loss {
public:
	void evaluate(const Matric& pred, const Matric& target);
	// 计算损失 和 下游梯度
};

#endif  // SRC_LOSS_MSE_LOSS_H_
