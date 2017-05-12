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
package com.example.android.renderscript_neuralnet;

/**
 * Created by miaowang on 8/15/16.
 */
public class ConvolveUtil {

    public static int get_conv_outsize(int size, int k, int s, int p) {
        return (size + p * 2 - k) / s + 1;
    }

    public static int get_deconv_outsize(int size, int k, int s, int p) {
        return s * (size - 1) + k - 2 * p;
    }

    public static float[] padd(float[] img, int img_channel,
                               int img_h, int img_w,
                               int pad_h, int pad_w) {
        int padded_h = img_h + 2 * pad_h;
        int padded_w = img_w + 2 * pad_w;

        float[] out = new float[padded_h * padded_w * img_channel];
        for (int ic = 0; ic < img_channel; ic++) {
            for (int ih = 0; ih < img_h; ih++) {
                int i_start_y = (img_channel * img_h + ih) * img_w;
                int o_start_y = (img_channel * padded_h + ih + pad_h) * padded_w;
                System.arraycopy(img, i_start_y, out, o_start_y, img_w);
            }
        }
        return out;
    }

    public static float[] unpadd(float[] img_padded, int img_channel,
                                 int img_h, int img_w,
                                 int pad_h, int pad_w) {
        int padded_h = img_h + 2 * pad_h;
        int padded_w = img_w + 2 * pad_w;

        float[] out = new float[img_channel * img_h * img_w];
        for (int ic = 0; ic < img_channel; ic++) {
            for (int ih = 0; ih < img_h; ih++) {
                // copy the unpadded region back as the img.
                int i_pos = ((img_channel * img_h) + ih) * img_w;
                int p_pos = ((img_channel * padded_h) + ih + pad_h) * padded_w;
                System.arraycopy(img_padded, p_pos, out, i_pos, img_w);
            }
        }
        return out;
    }

    // convert a multi-channel img to a col:
    // (img_channel * kernel_h * kernel_w) * (out_h * out_w)
    //               X                             Y
    public static float[] im2col(float[] img,
                                 int img_h, int img_w, int img_channel,
                                 int kernel_h, int kernel_w,
                                 int step_y, int step_x,
                                 int pad_h, int pad_w) {
        int out_h = get_conv_outsize(img_h, kernel_h, step_y, pad_h);
        int out_w = get_conv_outsize(img_w, kernel_w, step_x, pad_w);

        int dim_x = img_channel * kernel_h * kernel_w;
        int dim_y = out_h * out_w;

        float[] img_padded = padd(img, img_channel, img_h, img_w, pad_h, pad_w);
        int h_padded = img_h + pad_h * 2;
        int w_padded = img_w + pad_w * 2;

        float[] col = new float[dim_x * dim_y];

        for (int ih = 0; ih < out_h; ih++) {
            int img_ih = ih * step_y;
            for (int iw = 0; iw < out_w; iw++) {
                int img_iw = iw * step_x;
                int pos_col_start = ih * out_w * dim_x + iw * dim_x;
                for (int ic = 0; ic < img_channel; ic++) {
                    int pos_img_start = ic * h_padded * w_padded + img_ih * w_padded + img_iw;
                    int pos_c_start = ic * kernel_h * kernel_w;
                    for (int ikh = 0; ikh < kernel_h; ikh++) {
                        int pos_col = pos_col_start + pos_c_start + ikh * kernel_w;
                        int pos_img = pos_img_start + ikh * w_padded;
                        System.arraycopy(img_padded, pos_img, col, pos_col, kernel_w);
                    }
                }

            }
        }

        return col;
    }

    public static float[] col2im(float[] col,
                                 int col_h, int col_w, int col_channel,
                                 int kernel_h, int kernel_w,
                                 int step_y, int step_x,
                                 int pad_h, int pad_w) {

        int img_h = get_deconv_outsize(col_h, kernel_h, step_y, pad_h);
        int img_w = get_deconv_outsize(col_w, kernel_w, step_x, pad_w);

        // Dimension of the input col matrix.
        int dim_x = col_channel * kernel_h * kernel_w;
        int dim_y = col_h * col_w;

        int h_padded = img_h + pad_h * 2;
        int w_padded = img_w + pad_w * 2;

        float[] img_padded = new float[col_channel * h_padded * w_padded];

        for (int ic = 0; ic < col_channel; ic++) {
            int img_start = ic * w_padded * w_padded;
            int col_start = ic * kernel_h * kernel_w * col_h * col_w;
            for (int ikh = 0; ikh < kernel_h; ikh++) {
                for (int ikw = 0; ikw < kernel_w; ikw++) {
                    for (int ih = 0; ih < col_h; ih++) {
                        // Iterate over the bigger dimensions to get better memory coalescence.
                        int img_ih = ih * step_y;
                        for (int iw = 0; iw < col_w; iw++) {
                            int img_iw = iw * step_x;

                            // pos of the image
                            int img_pos = img_start + (img_ih + ikh) * w_padded + img_iw + ikw;
                            // pos of the kernel
                            int col_pos = col_start + ((ikh * kernel_w + ikw) * col_h + ih) * col_w + iw;
                            img_padded[img_pos] += col[col_pos];
                        }
                    }
                }
            }
        }

        return unpadd(img_padded, col_channel, img_h, img_w, pad_h, pad_w);
    }

}
