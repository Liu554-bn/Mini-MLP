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

    Vector mean = bottom.rowwise().mean();
    // mean 3*1
    std::cout << "mean:\n" << mean << std::endl;

    Matric mean_2 = bottom.colwise() - mean;
    std::cout << "减均值:\n" << mean_2 << std::endl; 

    Matric mean_3 = (bottom.colwise() - mean).array().square();
    std::cout << "平方:\n" << mean_3 << std::endl; 

    Vector mean_4 = (bottom.colwise() - mean).array().square().rowwise().mean();
    std::cout << "方差:\n" << mean_4 << std::endl;

    Matric x_norm = (bottom.colwise() - mean);
    std::cout << "减均值:\n" << x_norm << std::endl;
   
    float eps = 1e-5;
    mean_4 = (mean_4.array() + eps).sqrt();
    std::cout << "分母:\n" << mean_4 << std::endl;

    Matric x_norm_1 = x_norm.array().colwise() / mean_4.array();
    std::cout << "归一化:\n" << x_norm_1 << std::endl;

    Vector gamma; // 权重w n*p
    Vector beta; // 偏置b p*1

    gamma.resize(3);
    beta.resize(3);

    set_normal_random(gamma.data(), gamma.size(), 0, 0.01);
    set_normal_random(beta.data(), beta.size(), 0, 0.01);

    std::cout << "gamma:\n" << gamma << std::endl;
    std::cout << "beta:\n" << beta << std::endl;
    
    
    Matric out = x_norm_1.array().colwise() * gamma.array();
    std::cout << "平移:\n" << out << std::endl;
    
    out = out.colwise() + beta;
    std::cout << "平移:\n" << out << std::endl;
    
    return 0;
}


