#include "pool_core_3.h"

#define max(a, b) ((a>b)?a:b)
#define min(a, b) ((a>b)?b:a)

void Pool_3(int_type CHin, int_type Hin, int_type Win,
          int_type Kx, int_type Ky, int_type mode,
          Dtype_f feature_in[], Dtype_f feature_out[]
)//mode: 0:MEAN, 1:MIN, 2:MAX
{

    int_type Hout, Wout;
    Wout = Win / Kx;
    Hout = Hin / Ky;

    Pool_ch:
    for (int c = 0; c < 16; c++) {
#pragma HLS UNROLL
        Pool_h:
        for (int i = 0; i < 4; i++) {
            Pool_w:
            for (int j = 0; j < 4; j++) {
#pragma HLS PIPELINE II=1
                Dtype_f sum;
                if (mode == 0)
                    sum = 0;
                else if (mode == 1)
                    sum = 99999999999999999;
                else
                    sum = -99999999999999999;
                Pool_ky:
                for (int ii = 0; ii < 2; ii++) {
#pragma HLS UNROLL
                    Pool_kx:
                    for (int jj = 0; jj < 2; jj++) {
#pragma HLS UNROLL
                        int_type h = i * Ky + ii;
                        int_type w = j * Kx + jj;
                        switch (mode) {
                            case 0: {
                                sum += feature_in[h * CHin * Win + w * CHin + c];
                                break;
                            }
                            case 1: {
                                sum = min(sum, feature_in[h * CHin * Win + w * CHin + c]);
                                break;
                            }
                            case 2: {
                                sum = max(sum, feature_in[h * CHin * Win + w * CHin + c]);
                                break;
                            }
                            default:
                                break;
                        }
                    }
                }
                if (mode == 0)
                    sum = sum / (Kx * Ky);
                feature_out[i * Wout * CHin + j * CHin + c] = sum;
            }
        }
    }
}
