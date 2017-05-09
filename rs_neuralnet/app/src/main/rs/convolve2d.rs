/*
* Copyright (C) 2016 The Android Open Source Project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma version(1)
#pragma rs java_package_name(com.example.android.renderscript_neuralnet)
#pragma rs_fp_relaxed

rs_allocation img_alloc, padded_alloc, beta_alloc;
int img_h, img_w, img_channel;
int step_y, step_x, kernel_h, kernel_w, pad_h, pad_w;
int outH, outW;

// for tiled im2col
int tile_h = 0;
int tile_num = 0;


float RS_KERNEL addBeta(float in, uint32_t x, uint32_t y) {
    float beta = rsGetElementAt_float(beta_alloc, y);
    return in + beta;
}

float RS_KERNEL zero(float in) {
    return 0.0f;
}

void padd() {
    int padded_h = img_h + 2 * pad_h;
    int padded_w = img_w + 2 * pad_w;

    for (int ic = 0; ic < img_channel; ic++) {
        for (int ih = 0; ih < img_h; ih++) {
            int srcXoff = ih * img_w;
            int dstXoff = (ih + pad_h) * padded_w + pad_w;
            for (int iw = 0; iw < img_w; iw++) {
                float value = rsGetElementAt_float(img_alloc, srcXoff + iw, ic);
                rsSetElementAt_float(padded_alloc, value, dstXoff + iw, ic);
            }
        }
    }
}

// Parallel Tiled im2col
float RS_KERNEL im2col(uint32_t x, uint32_t y) {
    // x : outW * outH
    // y : in_channel * kernel_h * kernel_w
    if (y >= img_channel * kernel_h * kernel_w) {
        return 0.0f;
    }
    int ih = x / outW;
    int iw = x - ih * outW;

    int ic = y / (kernel_h * kernel_w);
    int ikh = (y - ic * (kernel_h * kernel_w)) / kernel_w;
    int ikw = y - ic * (kernel_h * kernel_w) - ikh * kernel_w;

    int h_padded = img_h + pad_h * 2;
    int w_padded = img_w + pad_w * 2;

    int img_ih = (ih + outH * tile_num) * step_y;
    float value = rsGetElementAt_float(padded_alloc, img_ih * w_padded + iw * step_x + ikh * w_padded + ikw, ic);
    return value;
}

// Reference implementation of convolve kernel.
// Performance not on par with (im2col + GEMM).
rs_allocation input_padded, W_alloc;
float RS_KERNEL convolve2D(uint32_t x, uint32_t y) {
    int h_padded = img_h + pad_h * 2;
    int w_padded = img_w + pad_w * 2;

    int ih = x / outW;
    int iw = x - outW * ih;
    float out = 0.0f;
    for (int ic = 0; ic < img_channel; ic++) {
        for (int ikh = 0; ikh < kernel_h; ikh++) {
            for (int ikw = 0; ikw < kernel_w; ikw++ ) {
                float cur_w = rsGetElementAt_float(W_alloc, ic * kernel_h * kernel_w + ikh * kernel_w + ikw, y);
                float cur_in = rsGetElementAt_float(input_padded, ih * step_y * w_padded + iw * step_x + ikh * w_padded + ikw, ic);
                out += cur_w * cur_in;
            }
        }
    }
    return out;
}
