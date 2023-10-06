/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 15:49:24
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 11:20:04
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

    Vector gamma; // 权重w n*p
    Vector beta; // 偏置b p*1

    gamma.resize(3);
    beta.resize(3);
    
    set_normal_random(gamma.data(), gamma.size(), 0, 0.01);
    set_normal_random(beta.data(), beta.size(), 0, 0.01);

    std::cout << "gamma:\n" << gamma << std::endl;
    std::cout << "beta:\n" << beta << std::endl;
    
    
    Vector mean = bottom.colwise().mean();
    std::cout << "mean:\n" << mean << std::endl;
    // mean 4*1
    Vector var =  (bottom.rowwise() - mean.transpose()).array().square().colwise().mean();
    std::cout << "var:\n" << var << std::endl;
    // var 4*1
    // Matric
    Matric x_halt = (bottom.rowwise() - mean.transpose()).array().rowwise() / (var.array() + eps).transpose();
    std::cout << "x_halt:\n" << x_halt << std::endl;

    Matric out = (x_halt.array().colwise() * gamma.array()).colwise() + beta.array();
    std::cout << "out:\n" << out << std::endl;
    // std n*1

    

    
    
    return 0;
}


