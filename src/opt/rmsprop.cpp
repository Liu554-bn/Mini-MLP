
#include "rmsprop.h"

void RMSProp::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw) {
    Vector& s = v_map[dw.data()];

    if (s.size() == 0) {
        s.resize(dw.size());
        s.setZero();
    }

    // Update s
    s = beta * s + (1 - beta) * dw.array().square().matrix();

    // Update w
    w -= lr * (dw.array() / (s.array().sqrt() + epsilon)).matrix();
}