/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-05-19 15:18:00
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-05-19 15:18:12
 */
#include <iostream>
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 
using namespace Eigen;

int main() {
    MatrixXf mat(3, 3);
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;
    
    float max_value;
    MatrixXf::Index max_index;
    
    max_value = mat.col(0).maxCoeff(&max_index);

    std::cout << "最大值为: " << max_value << std::endl;
    std::cout << "最大值所在的索引位置为: " << max_index << std::endl;

    return 0;
}
