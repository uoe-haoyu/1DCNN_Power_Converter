#ifndef __CONV_CORE_H__
#define __CONV_CORE_H__


#include <ap_int.h>
#include <ap_int.h>

#include <hls_math.h>
#include <math.h>
#include <iostream>

#include "types.h"

using namespace std;



//void Conv(ap_uint<16> CHin,ap_uint<16> Hin,ap_uint<16> Win,ap_uint<16> CHout,
//		ap_uint<8> Kx,ap_uint<8> Ky,ap_uint<8> Sx,ap_uint<8> Sy,ap_uint<1> mode,ap_uint<1> relu_en,
//		Dtype_f feature_in[],Dtype_w W[],Dtype_w bias[],Dtype_f feature_out[]
//	);//mode: 0:VALID, 1:SAME

//void Conv(int_type CHin,int_type Hin,int_type Win,int_type CHout,
//          int_type Kx,int_type Ky,int_type Sx,int_type Sy,bool_type mode,bool_type relu_en,
//          Dtype_f feature_in[],Dtype_w W[],Dtype_w bias[],Dtype_f feature_out[]
//);//mode: 0:VALID, 1:SAME

void Conv(int_type CHin,int_type Hin,int_type  Win,int_type  CHout,
          int_type  Kx,int_type  Ky,int_type Sx,int_type Sy,bool_type mode,bool_type relu_en,
          input_type feature_in[1][36][1],Dtype_w W[1][3][1][4],Dtype_w bias[4],Dtype_f feature_out[1][36][4]
);//mode: 0:VALID, 1:SAME
#endif
