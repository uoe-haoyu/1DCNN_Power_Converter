//
// Created by s1805689 on 12/03/2024.
//

#ifndef CNN_NORMAL_CNNNET_H
#define CNN_NORMAL_CNNNET_H

#include "conv_core.h"
#include "conv_core_2.h"
#include "conv_core_3.h"
#include "pool_core.h"
#include "pool_core_2.h"
#include "pool_core_3.h"
#include "types.h"
#include <ap_fixed.h>

#include <hls_math.h>
#include <math.h>


// IO

#define IN_CH 15
#define OUT_CH 6


// FC 1
#define OUT_FC_1 36
#define IN_FC_1 15
#define IN_HEIGHT_FC_1 36
#define IN_WIDTH_FC_1 15


// CNN 1
#define IN_HEIGHT_CNN_1 1
#define IN_WIDTH_CNN_1 36
#define IN_CH_CNN_1 1

#define KERNEL_HEIGHT_CNN_1 1
#define KERNEL_WIDTH_CNN_1 3
#define X_STRIDE_CNN_1 1
#define Y_STRIDE_CNN_1 1

#define RELU_EN_CNN_1  1
#define MODE_CNN_1     1          //0:VALID, 1:SAME
#define X_PADDING_CNN_1 (MODE_CNN_1?(KERNEL_WIDTH_CNN_1-1)/2:0)
#define Y_PADDING_CNN_1 (MODE_CNN_1?(KERNEL_HEIGHT_CNN_1-1)/2:0)

#define OUT_CH_CNN_1 4
#define OUT_WIDTH_CNN_1 ((IN_WIDTH_CNN_1+2*X_PADDING_CNN_1-KERNEL_WIDTH_CNN_1)/X_STRIDE_CNN_1+1)
#define OUT_HEIGHT_CNN_1 ((IN_HEIGHT_CNN_1+2*Y_PADDING_CNN_1-KERNEL_HEIGHT_CNN_1)/Y_STRIDE_CNN_1+1)


// POOL 1

#define MODE_POOL_1 2	//mode: 0:MEAN, 1:MIN, 2:MAX
#define IN_HEIGHT_POOL_1 1
#define IN_WIDTH_POOL_1 36
#define IN_CH_POOL_1 4

#define KERNEL_HEIGHT_POOL_1 1
#define KERNEL_WIDTH_POOL_1 2


#define OUT_WIDTH_POOL_1 (IN_WIDTH_POOL_1/KERNEL_WIDTH_POOL_1)
#define OUT_HEIGHT_POOL_1 (IN_HEIGHT_POOL_1/KERNEL_HEIGHT_POOL_1)


// attention1.pool1

#define MODE_POOL_2	0	//mode: 0:MEAN, 1:MIN, 2:MAX
#define IN_WIDTH_POOL_2 3
#define IN_HEIGHT_POOL_2 3
#define IN_CH_POOL_2 4

#define KERNEL_WIDTH_POOL_2 3
#define KERNEL_HEIGHT_POOL_2 3

#define OUT_WIDTH_POOL_2 (IN_WIDTH_POOL_2/KERNEL_WIDTH_POOL_2)
#define OUT_HEIGHT_POOL_2 (IN_HEIGHT_POOL_2/KERNEL_HEIGHT_POOL_2)


// attention1.conv1
#define IN_WIDTH_CNN_2 1
#define IN_HEIGHT_CNN_2 1
#define IN_CH_CNN_2 4

#define KERNEL_WIDTH_CNN_2 1
#define KERNEL_HEIGHT_CNN_2 1
#define X_STRIDE_CNN_2 1
#define Y_STRIDE_CNN_2 1

#define RELU_EN_CNN_2  1
#define MODE_CNN_2     1          //0:VALID, 1:SAME
#define X_PADDING_CNN_2 (MODE_CNN_2?(KERNEL_WIDTH_CNN_2-1)/2:0)
#define Y_PADDING_CNN_2 (MODE_CNN_2?(KERNEL_HEIGHT_CNN_2-1)/2:0)

#define OUT_CH_CNN_2 2
#define OUT_WIDTH_CNN_2 ((IN_WIDTH_CNN_2+2*X_PADDING_CNN_2-KERNEL_WIDTH_CNN_2)/X_STRIDE_CNN_2+1)
#define OUT_HEIGHT_CNN_2 ((IN_HEIGHT_CNN_2+2*Y_PADDING_CNN_2-KERNEL_HEIGHT_CNN_2)/Y_STRIDE_CNN_2+1)


// attention1.conv2
#define IN_WIDTH_CNN_3 1
#define IN_HEIGHT_CNN_3 1
#define IN_CH_CNN_3 2

#define KERNEL_WIDTH_CNN_3 1
#define KERNEL_HEIGHT_CNN_3 1
#define X_STRIDE_CNN_3 1
#define Y_STRIDE_CNN_3 1

#define RELU_EN_CNN_3  1
#define MODE_CNN_3     1          //0:VALID, 1:SAME
#define X_PADDING_CNN_3 (MODE_CNN_3?(KERNEL_WIDTH_CNN_3-1)/2:0)
#define Y_PADDING_CNN_3 (MODE_CNN_3?(KERNEL_HEIGHT_CNN_3-1)/2:0)

#define OUT_CH_CNN_3 4
#define OUT_WIDTH_CNN_3 ((IN_WIDTH_CNN_3+2*X_PADDING_CNN_3-KERNEL_WIDTH_CNN_3)/X_STRIDE_CNN_3+1)
#define OUT_HEIGHT_CNN_3 ((IN_HEIGHT_CNN_3+2*Y_PADDING_CNN_3-KERNEL_HEIGHT_CNN_3)/Y_STRIDE_CNN_3+1)


// FC 2
#define OUT_FC_2 32
#define IN_FC_2 36
#define IN_HEIGHT_FC_2 32
#define IN_WIDTH_FC_2 36

// FC 3
#define OUT_FC_SIGAMA 6
#define IN_FC_SIGAMA 32
#define IN_HEIGHT_FC_SIGAMA 6
#define IN_WIDTH_FC_SIGAMA 32



void cnn_forward(input_type input[15], output_type output[6]);

#endif //CNN_NORMAL_CNNNET_H


