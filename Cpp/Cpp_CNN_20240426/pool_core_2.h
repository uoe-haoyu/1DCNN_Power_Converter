#ifndef __POOL_CORE_2_H__
#define __POOL_CORE_2_H__


#include "types.h"
#include <ap_int.h>
#include <iostream>


//void Pool(ap_uint<16> CHin,ap_uint<16> Hin,ap_uint<16> Win,
//		ap_uint<8> Kx,ap_uint<8> Ky,ap_uint<2> mode,
//		Dtype_f feature_in[],Dtype_f feature_out[]
//	);



//void Pool_2(int_type CHin,int_type Hin,int_type Win,
//          int_type Kx,int_type Ky,int_type mode,
//          input_type feature_in[], output_type feature_out[]
//);//mode: 0:MEAN, 1:MIN, 2:MAX

void Pool_2(int_type CHin, int_type Hin, int_type Win,
            int_type Kx, int_type Ky, int_type mode,
            Dtype_f feature_in[3][3][4], Dtype_f feature_out[1][1][4]
);//mode: 0:MEAN, 1:MIN, 2:MAX

#endif
