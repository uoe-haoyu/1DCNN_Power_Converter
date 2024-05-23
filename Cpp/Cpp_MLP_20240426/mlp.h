#ifndef _MLP_H_
#define _MLP_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
//#include "math.h"
// ��������������������
//typedef double input_type;
//typedef double output_type;


typedef ap_fixed<32, 16> input_type;
typedef ap_fixed<32, 16> output_type;
//typedef ap_int<8> ap_int_4;
//typedef float input_type;
//typedef float output_type;

// ȫ���Ӳ�Ȩ�غ�ƫ�ã�ʾ�����ݣ�Ӧ�������ʵ��Ȩ�غ�ƫ�ý����滻��

void mlp_forward(input_type input[15], output_type output[6]);

#endif
