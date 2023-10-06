/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 10:59:40
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:31:22
 */
#ifndef SRC_LAYER_LAYERNORM_H_
#define SRC_LAYER_LAYERNORM_H_

#include <vector>
#include "..\layer.h"

class LayerNorm : public Layer {
private:
    const int dim_in; // 输入特征 n

    Vector gamma; // 权重w n*p
    Vector beta; // 偏置b p*1
    Vector grad_gamma; // 权重的梯度
    Vector grad_beta; // 偏置的梯度
    Vector mean;
    Vector var;
    float eps = 1e-5; // 防止除0的常数

    void init();

public:
    LayerNorm(const int dim_in) :
        dim_in(dim_in)
    {
        init();
    }

    void forward(const Matric& bottom);
    void backward(const Matric& bottom, const Matric& grad_top);
    void update(Optimizer& opt);
    int output_dim() { return dim_in; }
    std::vector<float> get_parameters() const;
    std::vector<float> get_derivatives() const;
    void set_parameters(const std::vector<float>& param);
};

#endif