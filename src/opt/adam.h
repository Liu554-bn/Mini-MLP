/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 22:14:45
 */
#ifndef SRC_OPTIMIZER_ADAM_H_
#define SRC_OPTIMIZER_ADAM_H_

#include <unordered_map>
#include "..\optimizer.h"

class Adam : public Optimizer {
	// 继承自 Optimizer
private:
    float lr, beta1, beta2, eps;
    int t;
    std::unordered_map<const float*, Vector> m_map, v_map;
	

public:
	explicit Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8) :
        lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {}
	
	void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};

#endif  // SRC_OPTIMIZER_SGD_H_
