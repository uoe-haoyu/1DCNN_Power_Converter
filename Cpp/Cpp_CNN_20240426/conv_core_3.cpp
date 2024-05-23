//
// Created by s1805689 on 15/03/2024.
//

#include "conv_core_3.h"


void Conv_3(int_type CHin, int_type Hin, int_type Win, int_type CHout,
            int_type Kx, int_type Ky, int_type Sx, int_type Sy, bool_type mode, bool_type relu_en,
            input_type feature_in[1][1][2], Dtype_w W[1][1][2][4], Dtype_w bias[4], Dtype_f feature_out[1][1][4]
)//mode: 0:VALID, 1:SAME
{
    int_type pad_x, pad_y;
    if (mode == 0) {
        pad_x = 0;
        pad_y = 0;
    } else {
        pad_x = 0;
        pad_y = 0;
    }
    int_type Hout, Wout;
    Wout = 1;
    Hout = 1;

//    input_type gamma[16] = {1.2046, 1.5192, 1.4257, 1.4594, 1.4671, 1.4191, 1.4722, 1.4946, 1.4364,
//                            1.3566, 1.2606, 1.3811, 1.5048, 1.3661, 1.5235, 1.5243};
//    input_type beta[16] = {0.0054, -0.0650, -0.0004,  0.1119,  0.0930,  0.1471,  0.0171,  0.1608,
//                           0.0546,  0.0885, -0.0099,  0.1426,  0.0237,  0.0171,  0.0285,  0.0943};
//    input_type epsilon = 0.0001;
//    input_type mean[16] = {-0.3405,  0.3536,  0.2874,  0.1339,  0.6020, -0.4168,  0.7333, -0.2133,
//                           -0.0372,  0.6128,  0.3096, -0.6116,  0.3168, -1.1240, -0.2484,  0.1584};
//    input_type variance[16] = {0.1250, 0.2365, 0.1426, 0.1774, 0.2351, 0.1488, 0.1611, 0.2503, 0.2277,
//                               0.2479, 0.1547, 0.2002, 0.1669, 0.3031, 0.1871, 0.1479};


    input_type conv2_out_temp[1][1][4] = {};
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv2_out_temp

    for (int cout = 0; cout < 4; cout++)
#pragma HLS UNROLL factor=2
            for (int i = 0; i < 1; i++)
                for (int j = 0; j < 1; j++) {
#pragma HLS PIPELINE II=1
                    Dtype_acc sum = 0;
                    for (int ii = 0; ii < 1; ii++)
                        for (int jj = 0; jj < 1; jj++) {
#pragma HLS UNROLL
                            int_type h = i * Sy - pad_y + ii;
                            int_type w = j * Sx - pad_x + jj;
                            if (h >= 0 && w >= 0 && h < 1 && w < 1) {
                                for (int cin = 0; cin < 2; cin++) {
#pragma HLS UNROLL
                                    //Feature [H][W][C]
                                    //kernel: [Ky][Kx][CHin][CHout]
                                    //Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
                                    //std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
                                    //std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
//                                    sum += feature_in[h * CHin * Win + w * CHin + cin] *
//                                           W[ii * Kx * CHin * CHout + jj * CHin * CHout + cin * CHout + cout];
                                    sum += feature_in[h][w][cin]*W[ii][jj][cin][cout];

                                }
                            }
                        }

                    sum += bias[cout];
                    conv2_out_temp[i][j][cout] = sum;

                }

//    input_type temp[256] = {};
//#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=temp

    for (int cout = 0; cout < 4; cout++)
//#pragma HLS UNROLL
        for (int i = 0; i < 1; i++)
//#pragma HLS UNROLL
            for (int j = 0; j < 1; j++) {
//#pragma HLS UNROLL
                output_type constant = 1;
        feature_out[i][j][cout] = constant / (1 + hls::exp(ap_fixed<32, 16>(-conv2_out_temp[i][j][cout])));
//        feature_out[i][j][cout] = constant / (1 + exp((-conv2_out_temp[i][j][cout])));
//                feature_out[i * Wout * CHout + j * CHout +cout] = constant / (1 + exp(-conv2_out_temp[i * Wout * CHout + j * CHout +cout]));
            }

}

//
//void Conv_3(int_type CHin, int_type Hin, int_type Win, int_type CHout,
//            int_type Kx, int_type Ky, int_type Sx, int_type Sy, bool_type mode, bool_type relu_en,
//            input_type feature_in[], Dtype_w W[], Dtype_w bias[], Dtype_f feature_out[]
//)//mode: 0:VALID, 1:SAME
//{
//    int_type pad_x, pad_y;
//    if (mode == 0) {
//        pad_x = 0;
//        pad_y = 0;
//    } else {
//        pad_x = (Kx - 1) / 2;
//        pad_y = (Ky - 1) / 2;
//    }
//    int_type Hout, Wout;
//    Wout = (Win + 2 * pad_x - Kx) / Sx + 1;
//    Hout = (Hin + 2 * pad_y - Ky) / Sy + 1;
//
////    input_type gamma[16] = {1.2046, 1.5192, 1.4257, 1.4594, 1.4671, 1.4191, 1.4722, 1.4946, 1.4364,
////                            1.3566, 1.2606, 1.3811, 1.5048, 1.3661, 1.5235, 1.5243};
////    input_type beta[16] = {0.0054, -0.0650, -0.0004,  0.1119,  0.0930,  0.1471,  0.0171,  0.1608,
////                           0.0546,  0.0885, -0.0099,  0.1426,  0.0237,  0.0171,  0.0285,  0.0943};
////    input_type epsilon = 0.0001;
////    input_type mean[16] = {-0.3405,  0.3536,  0.2874,  0.1339,  0.6020, -0.4168,  0.7333, -0.2133,
////                           -0.0372,  0.6128,  0.3096, -0.6116,  0.3168, -1.1240, -0.2484,  0.1584};
////    input_type variance[16] = {0.1250, 0.2365, 0.1426, 0.1774, 0.2351, 0.1488, 0.1611, 0.2503, 0.2277,
////                               0.2479, 0.1547, 0.2002, 0.1669, 0.3031, 0.1871, 0.1479};
//
//
//    input_type conv2_out_temp[4] = {};
//#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv2_out_temp
//
//    for (int cout = 0; cout < 4; cout++)
//#pragma HLS UNROLL factor=2
//            for (int i = 0; i < 1; i++)
//                for (int j = 0; j < 1; j++) {
//#pragma HLS PIPELINE II=1
//                    Dtype_acc sum = 0;
//                    for (int ii = 0; ii < 1; ii++)
//                        for (int jj = 0; jj < 1; jj++) {
//#pragma HLS UNROLL
//                            int_type h = i * Sy - pad_y + ii;
//                            int_type w = j * Sx - pad_x + jj;
//                            if (h >= 0 && w >= 0 && h < Hin && w < Win) {
//                                for (int cin = 0; cin < 2; cin++) {
//#pragma HLS UNROLL
//                                    //Feature [H][W][C]
//                                    //kernel: [Ky][Kx][CHin][CHout]
//                                    //Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
//                                    //std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
//                                    //std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
//                                    sum += feature_in[h * CHin * Win + w * CHin + cin] *
//                                           W[ii * Kx * CHin * CHout + jj * CHin * CHout + cin * CHout + cout];
//
//                                }
//                            }
//                        }
//
//                    sum += bias[cout];
//                    conv2_out_temp[i * Wout * CHout + j * CHout + cout] = sum;
//
//                }
//
////    input_type temp[256] = {};
////#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=temp
//
//    for (int cout = 0; cout < 4; cout++)
////#pragma HLS UNROLL
//        for (int i = 0; i < 1; i++)
////#pragma HLS UNROLL
//            for (int j = 0; j < 1; j++) {
////#pragma HLS UNROLL
//                output_type constant = 1;
//                feature_out[i * Wout * CHout + j * CHout +cout] = constant / (1 + hls::exp(ap_fixed<32, 16>(-conv2_out_temp[i * Wout * CHout + j * CHout +cout])));
////                feature_out[i * Wout * CHout + j * CHout +cout] = constant / (1 + exp(-conv2_out_temp[i * Wout * CHout + j * CHout +cout]));
//            }
//
//}
