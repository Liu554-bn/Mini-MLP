/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 15:49:24
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-29 20:56:17
 */
#include <iostream>
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matric;


int main() {
    float eps = 1e-5; // 防止除0的常数
    int n = 4;
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
    top << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;
    std::cout << "botom:\n" << botom << std::endl;

    Vector gamma(3);
    gamma << 1,1,1;
    std::cout << "gamma:\n" << gamma << std::endl;
    
    Vector beta(3);
    beta << 1,1,1;
    std::cout << "beta:\n" << beta << std::endl;

    Vector mean(3);
    mean << 1,1,1;
    std::cout << "mean:\n" << mean << std::endl;

    Vector val(3);
    val << 1,1,1;
    std::cout << "val:\n" << val << std::endl;

    Vector dbeta = grad_top.rowwise().sum();
    std::cout << "dbeta:\n" << dbeta << std::endl;

    Vector dgamma = top.cwiseProduct(grad_top).rowwise().sum();
    std::cout << "dgamma:\n" << dgamma << std::endl;
     
    Matric dx_hat = grad_top.array().colwise() * gamma.array();
    // dx_hat n*m
    std::cout << "dx_hat:\n" << dx_hat << std::endl;
    // 广播机制只能给vector用

    //Vector dsigma 

    Vector dsigma_one = (val.array() + eps).pow(-1.5);
    std::cout << "dsigma_one:\n" << dsigma_one << std::endl;

    Vector dsigma_two = -0.5 * dx_hat.cwiseProduct(dx_hat.colwise() - mean).rowwise().sum();
    std::cout << "dsigma_two:\n" << dsigma_two << std::endl;
    
    Vector dsigma = dsigma_two.cwiseProduct(dsigma_one);
    std::cout << "dsigma:\n" << dsigma << std::endl;

    
    //Vector dmu

    Vector dmu_one = 2 * dsigma.cwiseProduct((botom.colwise() - mean).rowwise().sum()) / n;
    std::cout << "dmu_one:\n" << dmu_one << std::endl;

    Vector dmu_two = -1 * (dx_hat.array().colwise() / (val.array() + eps).sqrt()).rowwise().sum();
    std::cout << "dmu_two:\n" << dmu_two << std::endl;

    Vector dmu =  dmu_two - dmu_one;
    std::cout << "dmu:\n" << dmu << std::endl;
    
    
    //Matric dx
    Matric dx_one = dx_hat.array().colwise() / (val.array() + eps).sqrt();
    std::cout << "dx_one:\n" << dx_one << std::endl;

    Matric dx_two = 2 * ((botom.colwise() - mean).array().colwise() * dsigma.array()) / n;
    std::cout << "dx_two:\n" << dx_two << std::endl;

    Vector dx_three = dmu.array() / n;
    std::cout << "dx_three:\n" << dx_three << std::endl;
    
    Matric dx = (dx_one + dx_two).colwise() + dx_three;
    std::cout << "dx:\n" << dx << std::endl;
    return 0;
}


