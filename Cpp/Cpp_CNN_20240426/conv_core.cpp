#include "conv_core.h"


//Feature: [H][W][C]
//kernel: [Ky][Kx][CHin][CHout]

void Conv(int_type CHin,int_type Hin,int_type  Win,int_type  CHout,
          int_type  Kx,int_type  Ky,int_type Sx,int_type Sy,bool_type mode,bool_type relu_en,
          input_type feature_in[1][36][1],Dtype_w W[1][3][1][4],Dtype_w bias[4],Dtype_f feature_out[1][36][4]
)//mode: 0:VALID, 1:SAME
{
    int_type pad_x, pad_y;
    if (mode == 0) {
        pad_x = 0;
        pad_y = 0;
    } else {
        pad_x = 1;
        pad_y = 0;
    }
    int_type Hout, Wout;
    Wout = 6;
    Hout = 6;

// MODIFY gamma_BN1 BEGIN
input_type gamma[4] = {2.7324, 2.3242, 2.7402, 2.5977};
// MODIFY gamma_BN1 END
// MODIFY beta_BN1 BEGIN
input_type beta[4] = {1.0566, 0.4915, 0.9854, 0.0364};
// MODIFY beta_BN1 END
    input_type epsilon = 0.0001;
// MODIFY mean_BN1 BEGIN
input_type mean[4] = { 0.0797, -0.0786,  0.1164,  0.0514};
// MODIFY mean_BN1 END
// MODIFY variance_BN1 BEGIN
input_type variance[4] = {0.0089, 0.0175, 0.0195, 0.0026};
// MODIFY variance_BN1 END

    input_type conv1_out_temp[1][36][4] = {};
//    input_type conv1_out_temp[144] = {};
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv1_out_temp

    for (int cout = 0; cout < 4; cout++){
#pragma HLS UNROLL
            for (int i = 0; i < 1; i++){
                for (int j = 0; j < 36; j++) {
#pragma HLS PIPELINE II=1
                    Dtype_acc sum = 0;
                    for (int ii = 0; ii < 1; ii++)
                        for (int jj = 0; jj < 3; jj++) {
#pragma HLS UNROLL
                            int_type h = i * Sy - pad_y + ii;
                            int_type w = j * Sx - pad_x + jj;
                            if (h >= 0 && w >= 0 && h < 1 && w < 36) {
                                for (int cin = 0; cin < 1; cin++) {
#pragma HLS UNROLL
                                    //Feature [H][W][C]
                                    //kernel: [Ky][Kx][CHin][CHout]
                                    //Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
                                    //std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
                                    //std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
//                        	sum +=feature_in[h*CHin*Win+w*CHin+cin]*W[ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout];
                                    sum += feature_in[h][w][cin] * W[ii][jj][cin][cout];
                                }
                            }
                        }

                    sum += bias[cout];
                    conv1_out_temp[i][j][cout] = sum;
//                    std::cout << "conv1_out_temp " << conv1_out_temp[i][j][cout] << std::endl;
                }
            }
}

    input_type temp[1][36][4] = {};
//    input_type temp[144] = {};
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=temp

    Conv_BN_label0:for(int cout=0;cout<4;cout++)
#pragma HLS UNROLL
	   Conv_BN_label1:for(int i=0;i<1;i++)

//#pragma HLS UNROLL
	       Conv_BN_label2:for(int j=0;j<36;j++) {
//#pragma HLS UNROLL
                // 归一化�?缩放�?平移


                temp[i][j][cout] = gamma[cout] * (conv1_out_temp[i][j][cout] - mean[cout]) / hls::sqrt(variance[cout] + epsilon) + beta[cout];
//                temp[i][j][cout] = gamma[cout] * (conv1_out_temp[i][j][cout] - mean[cout]) / sqrt(variance[cout] + epsilon) + beta[cout];
//                std::cout << "BN" << temp[i][j][cout] << endl;
                if (relu_en & (temp[i][j][cout] < 0))
                    feature_out[i][j][cout] = 0;
                    //feature_out[i][j][cout]=sum;
                else { feature_out[i][j][cout] = temp[i][j][cout];}
//                std::cout << "Relu" << feature_out[i][j][cout] << endl;
            }

}







//
//void Conv(int_type CHin,int_type Hin,int_type  Win,int_type  CHout,
//          int_type  Kx,int_type  Ky,int_type Sx,int_type Sy,bool_type mode,bool_type relu_en,
//          input_type feature_in[],Dtype_w W[],Dtype_w bias[],Dtype_f feature_out[]
//)//mode: 0:VALID, 1:SAME
//{
//    int_type pad_x,pad_y;
//    if(mode==0)
//    {
//        pad_x=0;pad_y=0;
//    }
//    else
//    {
//        pad_x=(Kx-1)/2;
//        pad_y=(Ky-1)/2;
//    }
//    int_type Hout,Wout;
//    Wout=(Win+2*pad_x-Kx)/Sx+1;
//    Hout=(Hin+2*pad_y-Ky)/Sy+1;
//
//    input_type gamma[4] = {1.6714, 1.3761, 2.7516, 1.8736};
//    input_type beta[4] = {0.1439, -0.0730, -0.2548, -0.3513};
//    input_type epsilon = 0.0001;
//    input_type mean[4] = { 0.0251,  0.0846,  0.1640, -0.1039};
//    input_type variance[4] = {0.0259, 0.0194, 0.0238, 0.0332};
//
//
////    input_type conv1_out_temp[6][6][4] = {};
//    input_type conv1_out_temp[144] = {};
//#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv1_out_temp
//
//    for(int cout=0;cout<4;cout++)
//#pragma HLS UNROLL
//            for(int i=0;i<6;i++)
//                for(int j=0;j<6;j++)
//                {
//#pragma HLS PIPELINE II=1
//                    Dtype_acc sum=0;
//                    for(int ii=0;ii<3;ii++)
//                        for(int jj=0;jj<3;jj++)
//                        {
//#pragma HLS UNROLL
//                            int_type h=i*Sy-pad_y+ii;
//                            int_type w=j*Sx-pad_x+jj;
//                            if(h>=0 && w>=0 && h<Hin && w<Win)
//                            {
//                                for(int cin=0;cin<1;cin++)
//                                {
//#pragma HLS UNROLL
//                                    //Feature [H][W][C]
//                                    //kernel: [Ky][Kx][CHin][CHout]
//                                    //Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
//                                    //std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
//                                    //std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
//                                    sum +=feature_in[h*CHin*Win+w*CHin+cin]*W[ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout];
////                                sum += feature_in[h][w][cin]*w[ii][jj][cin][cout];
//                                }
//                            }
//                        }
//
//                    sum+=bias[cout];
//                    conv1_out_temp[i*Wout*CHout+j*CHout+cout]=sum;
//
//                }
//
////    input_type temp[6][6][4] = {};
//    input_type temp[144] = {};
//#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=temp
//
//    Conv_BN_label0:for(int cout=0;cout<4;cout++)
//#pragma HLS UNROLL
//    Conv_BN_label1:for(int i=0;i<6;i++)
//
////#pragma HLS UNROLL
//    Conv_BN_label2:for(int j=0;j<6;j++) {
////#pragma HLS UNROLL
//    // 归一化�?缩放�?平移
//
////                int idx = i * Wout * CHout + j * CHout + cout;
//    temp[i * Wout * CHout + j * CHout + cout] = gamma[cout] * (conv1_out_temp[i * Wout * CHout + j * CHout + cout] - mean[cout]) / hls::sqrt(variance[cout] + epsilon) + beta[cout];
////                temp[i * Wout * CHout + j * CHout + cout] = gamma[cout] * (conv1_out_temp[i * Wout * CHout + j * CHout + cout] - mean[cout]) / sqrt(variance[cout] + epsilon) + beta[cout];
////                cout << temp << endl;
//    if (relu_en & (temp[i * Wout * CHout + j * CHout + cout] < 0))
//        feature_out[i * Wout * CHout + j * CHout + cout] = 0;
//        //feature_out[i][j][cout]=sum;
//    else { feature_out[i * Wout * CHout + j * CHout + cout] = temp[i * Wout * CHout + j * CHout + cout];}
//}
//
//}
//
//





