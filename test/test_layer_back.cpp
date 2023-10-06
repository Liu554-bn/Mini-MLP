/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 15:49:24
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 12:08:51
 */
#include <iostream>
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core"  

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matric;


int main() {
    float eps = 1e-5; // 防止除0的常数
    int n = 3;
    Eigen::MatrixXf grad_top(3, 4);
    grad_top << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;
    std::cout << "grad_top:\n" << grad_top << std::endl;

    Eigen::MatrixXf top(3, 4);
    top << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;
    std::cout << "top:\n" << top << std::endl;

    Eigen::MatrixXf botom(3, 4);
    botom << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;
    std::cout << "botom:\n" << botom << std::endl;

    Vector gamma(3);
    gamma << 1,1,1;
    std::cout << "gamma:\n" << gamma << std::endl;
    
    Vector beta(3);
    beta << 1,1,1;
    std::cout << "beta:\n" << beta << std::endl;

    Vector mean(4);
    mean << 1,1,1,1;
    std::cout << "mean:\n" << mean << std::endl;

    Vector val(4);
    val << 1,1,1,1;
    std::cout << "val:\n" << val << std::endl;

    Vector dbeta = grad_top.rowwise().sum();
    std::cout << "dbeta:\n" << dbeta << std::endl;

    Vector dgamma = top.cwiseProduct(grad_top).rowwise().sum();
    std::cout << "dgamma:\n" << dgamma << std::endl;
     
    Matric dx_hat = grad_top.array().colwise() * gamma.array();
    // dx_hat n*m
    std::cout << "dx_hat:\n" << dx_hat << std::endl;
    // 广播机制只能给vector用

    Vector dsigma_one = -0.5 * (botom.rowwise() - mean.transpose()).colwise().sum();
    std::cout << "dsigma_one:\n" << dsigma_one << std::endl;

    Vector dsigma_two = (val.array() + eps).pow(-1.5);
    std::cout << "dsigma_two:\n" << dsigma_two << std::endl;
    
    Vector dsigma = dsigma_one.cwiseProduct(dsigma_two);
    std::cout << "dsigma:\n" << dsigma << std::endl;

    Vector dmu_one = -1 * (dx_hat.array().rowwise() / (val.array() + eps).sqrt().transpose()).colwise().sum();
    std::cout << "dmu_one:\n" << dmu_one << std::endl;

    Vector dmu_two = 2 * dsigma.cwiseProduct((botom.rowwise() - mean.transpose()).colwise().sum().transpose()) / n;
    std::cout << "dmu_two:\n" << dmu_two << std::endl;

    Vector dmu =  dmu_one - dmu_two;
    std::cout << "dmu:\n" << dmu << std::endl;

    Matric dx_one = dx_hat.array().rowwise() / (val.array() + eps).sqrt().transpose();
    std::cout << "dx_one:\n" << dx_one << std::endl;

    Matric dx_two = 2 * ((botom.rowwise() - mean.transpose()).array().rowwise() * dsigma.array().transpose()) / n;
    std::cout << "dx_two:\n" << dx_two << std::endl;

    Vector dx_three = dmu.array() / n;
    std::cout << "dx_three:\n" << dx_three << std::endl;
    
    Matric dx = (dx_one + dx_two).rowwise() + dx_three.transpose();
    std::cout << "dx:\n" << dx << std::endl;
    return 0;
}


