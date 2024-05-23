//
// Created by s1805689 on 15/03/2024.
//

#include <ap_int.h>
#include <ap_int.h>

#include <hls_math.h>
#include <math.h>
#include <iostream>

#include "types.h"
using namespace std;


#ifndef CNN_NORMAL_CONV_CORE_2_H
#define CNN_NORMAL_CONV_CORE_2_H


//void Conv_2(int_type CHin,int_type Hin,int_type Win,int_type CHout,
//          int_type Kx,int_type Ky,int_type Sx,int_type Sy,bool_type mode,bool_type relu_en,
//          Dtype_f feature_in[],Dtype_w W[],Dtype_w bias[],Dtype_f feature_out[]
//);//mode: 0:VALID, 1:SAME


void Conv_2(int_type CHin, int_type Hin, int_type Win, int_type CHout,
            int_type Kx, int_type Ky, int_type Sx, int_type Sy, bool_type mode, bool_type relu_en,
            input_type feature_in[1][1][4], Dtype_w W[1][1][4][2], Dtype_w bias[2], Dtype_f feature_out[1][1][2]
);//mode: 0:VALID, 1:SAME

#endif //CNN_NORMAL_CONV_CORE_2_H
