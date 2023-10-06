/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 18:59:45
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 21:07:19
 */
#include "sgd.h"

void SGD::update(Vector::AlignedMapType& w,Vector::ConstAlignedMapType& dw) {
	
	Vector& v = v_map[dw.data()];
	// dw.data() 浮点型数组 或者 浮点型指针
	// 这个东西，每个层的参数都有
	if (v.size() == 0) {
		v.resize(dw.size());
		v.setZero();
	}// 初始化动态数组 初值为0
	// update v
	v = lr*dw;
	// update w
	w -= v;
}
