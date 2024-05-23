//
// Created by s1805689 on 12/03/2024.
//

#ifndef CNN_NORMAL_TYPES_H
#define CNN_NORMAL_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>


//typedef bool bool_type;
//typedef ap_int<32> int_type;
//
//typedef ap_fixed<32, 16> input_type;
//typedef ap_fixed<32, 16> output_type;
//
//typedef ap_fixed<32, 16> Dtype_f;
//typedef ap_fixed<32, 16> Dtype_w;
//
//typedef ap_fixed<32, 16> Dtype_mul;
//typedef ap_fixed<32, 16> Dtype_acc;


typedef bool bool_type;
typedef ap_int<20> int_type;

typedef ap_fixed<20, 10> input_type;
typedef ap_fixed<20, 10> output_type;

typedef ap_fixed<20, 10> Dtype_f;
typedef ap_fixed<20, 10> Dtype_w;
typedef ap_fixed<20, 10> Dtype_mul;
typedef ap_fixed<20, 10> Dtype_acc;



//typedef bool bool_type;
//typedef int int_type;
//
//typedef float input_type;
//typedef float output_type;
//
//typedef float Dtype_f;
//typedef float Dtype_w;
//typedef float Dtype_mul;
//typedef float Dtype_acc;


#endif //CNN_NORMAL_TYPES_H
