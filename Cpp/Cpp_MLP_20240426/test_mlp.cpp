#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"

int main() {
    // ����ʾ������
//    input_type input[15] = {100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    input_type input[15] = {84.89 ,-49.13, 149.30 ,-86.41, 86.97 ,-23.81 ,-49.64, 0.24, 2.16 ,0.00, 0.00 ,0.00 ,1.00 ,0.00, 0.00};
// 0 0 1 1 0 0

    input_type input_n[15] = {1.82297479, -1.06411679, 1.22065991, -0.71148063, 1.86752779, -4.12793444, -1.07879153,
                              0.05047604, 0.53996643, 0., 0., 0., 1., 0., 0.};


    output_type output[6] = {0,0,0,0,0,0};

    for (int i = 0; i < 6; i++) {
        std::cout << input_n[i] << " input_n " << std::endl;
    }

//    std::cout << hls::exp(ap_fixed<32,24>(2)) << std::endl;

    // ���з���
    mlp_forward(input, output);

    // ��ӡ���
    printf("Output: ");
    for (int i = 0; i < 6; i++) {
        std::cout <<  output[i] << std::endl;
    }
    printf("\n");

    return 0;
}
