/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 21:55:18
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-28 22:28:25
 */
#include "adam.h"

void Adam::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw) {
    Vector& m = m_map[dw.data()];
    Vector& v = v_map[dw.data()];

    // Initialize moving averages
    if (m.size() == 0) {
        m.resize(dw.size());
        m.setZero();
    }
    if (v.size() == 0) {
        v.resize(dw.size());
        v.setZero();
    }

    // Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * dw;

    // Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * dw.array().square().matrix();

    // Update parameter
    t++;

    Vector m_hat = m / (1 - std::pow(beta1, t));

    Vector v_hat = v / (1 - std::pow(beta2, t));

    w -= (lr * m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
}
