/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 15:49:24
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-29 16:00:01
 */
#include <iostream>
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matric;


int main() {

    Eigen::MatrixXf bottom(3, 4);
    bottom << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;
    std::cout << "bottom:\n" << bottom << std::endl;

    float eps = 1e-5;
    float momentum = 0.9;

    Vector gamma; // 权重w n*p
    Vector beta; // 偏置b p*1
    Vector running_mean;
    Vector running_var;

    gamma.resize(3);
    beta.resize(3);
    running_mean.resize(3);
    running_var.resize(3);
    std::cout << "平移:\n" << running_mean << std::endl;
    set_normal_random(gamma.data(), gamma.size(), 0, 0.01);
    set_normal_random(beta.data(), beta.size(), 0, 0.01);

    std::cout << "gamma:\n" << gamma << std::endl;
    std::cout << "beta:\n" << beta << std::endl;
    
    
    Vector mean = bottom.rowwise().mean();
    // mean n*4
    Vector var = (bottom.colwise() - mean).array().square().rowwise().mean();
    // std n*1

    running_mean = momentum * running_mean + (1 - momentum) * mean;
    running_var = momentum * running_var + (1 - momentum) * var;
    
    Matric x_norm = (bottom.colwise() - running_mean);
    // x_norm m*m
    running_var = (running_var.array() + eps).sqrt();
    // std n*1
    Matric x_norm_1 = x_norm.array().colwise() / running_var.array();
    // x_norm_1 n*m

    Matric out = x_norm_1.array().colwise() * gamma.array();
    out = out.colwise() + beta;
    std::cout << "平移:\n" << out << std::endl;
    
    
    return 0;
}


