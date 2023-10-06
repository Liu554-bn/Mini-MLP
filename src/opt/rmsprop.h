/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-17 15:29:31
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 21:35:40
 */
#ifndef SRC_OPTIMIZER_RMS_H_
#define SRC_OPTIMIZER_RMS_H_

#include <unordered_map>
#include "..\optimizer.h"

class RMSProp : public Optimizer {
	// 继承自 Optimizer
private:
	float beta;  
    float epsilon;  
	std::unordered_map<const float*, Vector> v_map;
	

public:
	explicit RMSProp(float lr = 0.01, float beta = 0.999, float epsilon = 1e-8) 
        : Optimizer(lr),beta(beta),epsilon(epsilon) {}
	
	void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};

#endif  // SRC_OPTIMIZER_SGD_H_
