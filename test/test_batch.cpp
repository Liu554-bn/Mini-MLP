/*
 * @Descripttion: 
 * @version: 
 * @Author: BaoBaBu
 * @Date: 2023-04-29 11:31:29
 * @LastEditors: BaoBaBu
 * @LastEditTime: 2023-04-30 15:51:14
 */

#include "..\src\layer\batch_norm.h"
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 

void test() {
  const int n = 3;
  const int p = 2;
  const float eps = 1e-6;

  // 输入数据
  Matric bottom(n, p);
  bottom << 1, 2, 3, 4, 5, 6;

  // 期望的输出数据
  Matric expected_top(p, n);
  expected_top << -0.999995, -0.999983, -0.999970,
                  0.999995,  0.999983,  0.999970;

  // 创建 BatchNorm 层
  BatchNorm layer(p);

  // 前向传播
  layer.forward(bottom);


}
