#include "pool_core.h"

#define max(a, b) ((a>b)?a:b)
#define min(a, b) ((a>b)?b:a)

void Pool(int_type CHin, int_type Hin, int_type Win,
          int_type Kx, int_type Ky, int_type mode,
          Dtype_f feature_in[1][36][4], Dtype_f feature_out[1][18][4]
)//mode: 0:MEAN, 1:MIN, 2:MAX
{

    int_type Hout, Wout;
    Wout = Win / Kx;
    Hout = Hin / Ky;

    Pool_ch:
    for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
        Pool_h:
        for (int i = 0; i < 1; i++) { //Hout
            Pool_w:
            for (int j = 0; j < 18; j++) { //Wout
#pragma HLS PIPELINE II=1
                Dtype_f sum;
                if (mode == 0)
                    sum = 0;
                else if (mode == 1)
                    sum = 500;
                else
                    sum = -500;
                Pool_ky:
                for (int ii = 0; ii < 1; ii++) {
#pragma HLS UNROLL
                    Pool_kx:
                    for (int jj = 0; jj < 2; jj++) {
#pragma HLS UNROLL
                        int_type h = i * Ky + ii;
                        int_type w = j * Kx + jj;
                        switch (mode) {
                            case 0: {
                                sum += feature_in[h][w][c];
                                break;
                            }
                            case 1: {
                                sum = min(sum, feature_in[h][w][c]);
                                break;
                            }
                            case 2: {
//                                std::cout << "sum be " << sum << std::endl;
//                                std::cout << "feature_in " << feature_in[i][j][c] << std::endl;
                                sum = max(sum, feature_in[h][w][c]);
//                                std::cout << "sum af " << sum << std::endl;
                                break;
                            }
                            default:
                                break;
                        }
                    }
                }
                if (mode == 0)
                    sum = sum / (Kx * Ky);
                feature_out[i][j][c] = sum;
//                std::cout << "pool " << feature_out[i][j][c] << std::endl;
            }
        }
    }
}
//
//void Pool(int_type CHin, int_type Hin, int_type Win,
//          int_type Kx, int_type Ky, int_type mode,
//          Dtype_f feature_in[], Dtype_f feature_out[]
//)//mode: 0:MEAN, 1:MIN, 2:MAX
//{
//
//    int_type Hout, Wout;
//    Wout = Win / Kx;
//    Hout = Hin / Ky;
//
//    Pool_ch:
//    for (int c = 0; c < 4; c++) {
//#pragma HLS UNROLL
//        Pool_h:
//        for (int i = 0; i < 3; i++) { //Hout
//            Pool_w:
//            for (int j = 0; j < 3; j++) { //Wout
//#pragma HLS PIPELINE II=1
//                Dtype_f sum;
//                if (mode == 0)
//                    sum = 0;
//                else if (mode == 1)
//                    sum = 30000;
//                else
//                    sum = -30000;
//                Pool_ky:
//                for (int ii = 0; ii < 2; ii++) {
//#pragma HLS UNROLL
//                    Pool_kx:
//                    for (int jj = 0; jj < 2; jj++) {
//#pragma HLS UNROLL
//                        int_type h = i * Ky + ii;
//                        int_type w = j * Kx + jj;
//                        switch (mode) {
//                            case 0: {
//                                sum += feature_in[h * CHin * Win + w * CHin + c];
//                                break;
//                            }
//                            case 1: {
//                                sum = min(sum, feature_in[h * CHin * Win + w * CHin + c]);
//                                break;
//                            }
//                            case 2: {
//                                sum = max(sum, feature_in[h * CHin * Win + w * CHin + c]);
//                                break;
//                            }
//                            default:
//                                break;
//                        }
//                    }
//                }
//                if (mode == 0)
//                    sum = sum / (Kx * Ky);
//                feature_out[i * Wout * CHin + j * CHin + c] = sum;
//            }
//        }
//    }
//}
