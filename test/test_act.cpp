/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-28 19:41:30
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:49:27
 */
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 
#include "..\src\act\leaky_relu.cpp"
#include "..\src\act\elu.cpp"
#include "..\src\act\tanh.cpp"

using namespace Eigen;

int main() {
    // 创建一个3x3的测试矩阵
    MatrixXf test_matrix(3, 3);
    test_matrix << -1, 2, -3,
                    4, -5, 6,
                    -7, 8, -9;
    // 创建Leaky ReLU实例
    Tanh leaky_relu;
    // 前向传播
    leaky_relu.forward(test_matrix);
    // 打印前向传播结果
    std::cout << "Leaky ReLU Forward:\n" << leaky_relu.top << std::endl;
    // 反向传播
    MatrixXf grad_top(3, 3);
    grad_top << 1, 2, 3,
                4, 5, 6,
                7, 8, 9;
    leaky_relu.backward(test_matrix, grad_top);
    // 打印反向传播结果
    std::cout << "Leaky ReLU Backward:\n" << leaky_relu.grad_bottom << std::endl;

    return 0;
}
