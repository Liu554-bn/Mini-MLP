#include "tanh.h"

void Tanh::forward(const Matric& bottom) {
    // 计算tanh函数
    top = bottom.array().tanh();
}

void Tanh::backward(const Matric& bottom, const Matric& grad_top) {
    grad_bottom = (1 - top.array().square()) * grad_top.array();
}
