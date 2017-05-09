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

rs_allocation img_alloc, col_alloc, padded_alloc, beta_alloc;
int img_h, img_w, img_channel;
int step_y, step_x, kernel_h, kernel_w, pad_h, pad_w;
int col_h, col_w, col_channel;

float RS_KERNEL addBeta(float in, uint32_t x, uint32_t y) {
    float beta = rsGetElementAt_float(beta_alloc, y);
    return in + beta;
}

float RS_KERNEL zero(float in) {
    return 0.0f;
}

void unpadd() {
    int padded_h = img_h + 2 * pad_h;
    int padded_w = img_w + 2 * pad_w;

    for (int ic = 0; ic < img_channel; ic++) {
        for (int ih = 0; ih < img_h; ih++) {
            int srcXoff = ih * img_w;
            int dstXoff = (ih + pad_h) * padded_w + pad_w;
            for (int iw = 0; iw < img_w; iw++) {
                float value = rsGetElementAt_float(padded_alloc, dstXoff + iw, ic);
                rsSetElementAt_float(img_alloc, value, srcXoff + iw, ic);
            }
        }
    }
}


static int get_deconv_outsize(int size, int k, int s, int p) {
    return s * (size - 1) + k - 2 * p;
}


// for tiled col2im
int tile_h = 0;
int tile_num = 0;

// Tiled col2im implementation
void col2im_tileY() {
    if (tile_h == 0) tile_h = col_h;

    int img_h = get_deconv_outsize(tile_h, kernel_h, step_y, pad_h);
    int img_w = get_deconv_outsize(col_w, kernel_w, step_x, pad_w);

    // Dimension of the input col matrix.
    int dim_x = col_channel * kernel_h * kernel_w;
    int dim_y = tile_h * col_w;

    int h_padded = img_h + pad_h * 2;
    int w_padded = img_w + pad_w * 2;

    for (int ic = 0; ic < col_channel; ic++) {
        int imgY = ic;
        int colYoff = ic * kernel_h * kernel_w;
        for (int ikh = 0; ikh < kernel_h; ikh ++) {
            for (int ikw = 0; ikw < kernel_w; ikw++) {
                int colY = colYoff + ikh * kernel_w + ikw;
                for (int ih = 0; ih < tile_h; ih++) {
                    // Iterate over the bigger dimensions to get better memory locality.
                    int imgXoff = ((ih + tile_h * tile_num) * step_y + ikh) * w_padded;
                    for (int iw = 0; iw < col_w; iw++) {
                        // pos of the image
                        int imgX = imgXoff + iw * step_x + ikw;
                        // pos of the kernel
                        int colX = ih * col_w + iw;

                        float col_value = rsGetElementAt_float(col_alloc, colX, colY);
                        float img_value = rsGetElementAt_float(padded_alloc, imgX, imgY) + col_value;
                        rsSetElementAt_float(padded_alloc, img_value, imgX, imgY);
                    }
                }
            }
        }
    }
}


// Parallel implementation of col2im.
// Performance not ideal on CPU.
float RS_KERNEL col2imPar(float in, uint32_t x, uint32_t y) {
  // x : h_padded * w_padded
  // y : col_channel
    float out = in;
    int h_padded = img_h + pad_h * 2;
    int w_padded = img_w + pad_w * 2;

    int temph = x / w_padded;
    int tempw = x - temph * w_padded;

    int colYoff = y * kernel_h * kernel_w;
    for (int ikh = 0; ikh < kernel_h; ikh ++) {
        int ih = temph - ikh;
        if ( ih % step_y == 0) {
            ih = ih / step_y;
            for (int ikw = 0; ikw < kernel_w; ikw++) {
                int iw = tempw - ikw;
                if (iw % step_x == 0) {
                    iw = iw / step_x;
                    int colY = colYoff + ikh * kernel_w + ikw;
                    int colX = ih * col_w + iw;
                    float col_value = rsGetElementAt_float(col_alloc, colX, colY);
                    out += col_value;
                }
            }
        }
    }
    return out;
}

// col2im reference implementation
void col2im() {
    int img_h = get_deconv_outsize(col_h, kernel_h, step_y, pad_h);
    int img_w = get_deconv_outsize(col_w, kernel_w, step_x, pad_w);

    // Dimension of the input col matrix.
    int dim_x = col_channel * kernel_h * kernel_w;
    int dim_y = col_h * col_w;

    int h_padded = img_h + pad_h * 2;
    int w_padded = img_w + pad_w * 2;

    for (int ic = 0; ic < col_channel; ic++) {
        int imgY = ic;
        int colYoff = ic * kernel_h * kernel_w;
        for (int ikh = 0; ikh < kernel_h; ikh ++) {
            for (int ikw = 0; ikw < kernel_w; ikw++) {
                int colY = colYoff + ikh * kernel_w + ikw;
                for (int ih = 0; ih < col_h; ih++) {
                    // Iterate over the bigger dimensions to get better memory coalescence.
                    int imgXoff = (ih * step_y + ikh) * w_padded;
                    for (int iw = 0; iw < col_w; iw++) {
                        // pos of the image
                        int imgX = imgXoff + iw * step_x + ikw;
                        // pos of the kernel
                        int colX = ih * col_w + iw;

                        float col_value = rsGetElementAt_float(col_alloc, colX, colY);
                        float img_value = rsGetElementAt_float(padded_alloc, imgX, imgY) + col_value;
                        rsSetElementAt_float(padded_alloc, img_value, imgX, imgY);
                    }
                }
            }
        }
    }
}
