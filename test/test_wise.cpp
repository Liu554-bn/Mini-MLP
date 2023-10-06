/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 15:49:24
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-29 20:17:06
 */
#include <iostream>
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matric;


int main() {
    Eigen::MatrixXf grad_top(3, 4);
    grad_top << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;
    std::cout << "grad_top:\n" << grad_top << std::endl;
  

    Vector zeta(3,1);
    zeta << 1,1,1;
    // 广播机制只能给vector 或者类型是列向量的矩阵用
    grad_top.colwise() +=  zeta;
    std::cout << "zeta:\n" << grad_top << std::endl;
  
    return 0;
}




