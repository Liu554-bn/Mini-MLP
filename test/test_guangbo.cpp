
#include <iostream>
#include "..\thirdParty\eigen-3.4.0\Eigen\Dense" 
#include "..\thirdParty\eigen-3.4.0\Eigen\Core" 

int main() {
    Eigen::MatrixXf bottom(3, 4);
    bottom << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;

    Eigen::VectorXf mean(3);
    mean << 2, 4, 6;

    bottom = bottom.colwise() - mean;

    std::cout << bottom << std::endl;
}

