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

import android.content.Context;
import android.content.res.AssetManager;
import android.support.v8.renderscript.Allocation;
import android.support.v8.renderscript.Element;
import android.support.v8.renderscript.RenderScript;
import android.support.v8.renderscript.ScriptIntrinsicBLAS;
import android.support.v8.renderscript.Type;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * Created by miaowang on 8/15/16.
 */


/*
    Two-dimensional chained Residual Block layer.
    Each Residual Block consists of 2 Convolution operations and two BatchNormalization operations.
    The calculated residual will be added with the input image.

    Unlike normal ResidualBlock, ResidualBlockChained tries to perform convolution and
    batch normalization in the same layer, which helps better manage and reuse intermediate
    Allocations, reducing memory pressure to the system and improving overall performance.

    Attributes:
    in_channels  :  Number of channels of input arrays.
    out_channels :  Number of channels of output arrays.
    mNumBlocks   :  Number if ResidualBlocks chained.
    ksize        :  Size of filters (a.k.a. kernels).
    stride       :  Stride of filter applications.
    pad          :  Spatial padding width for input arrays.
    W            :  Weight parameter for convolution.
    b            :  Bias parameter for convolution.
    gamma        :  Scaling parameter for batch normalization.
    beta         :  Shifting parameter for batch normalization.
    avg_mean     :  Population mean for batch normalization.
    avg_var      :  Population variance for batch normalization.


*/
public class ResidualBlockChained extends NeuralNetLayerBase {
    // The dimension in Y for each tile.
    private final int TILE_Y = 64;

    // The dimension of the image after ResidualBlock.
    // Used by subsequent operations (layers).
    public int outH, outW;

    // The padded dimension to satisfy alignment requirement for certain GPUs.
    private int padded_Y_blas;

    private int mNumBlocks;
    private int in_channels, out_channels;
    private int pad = 1;
    private int stride = 1;
    private int ksize = 3;

    private float[] W; // X : out_channels * (in_channels * ksize * ksize) : Y;
    private float[] b;

    private float[] gamma;
    private float[] beta;
    private float[] avg_mean;
    private float[] avg_var;

    private Allocation[] W_alloc, b_alloc;
    private Allocation[] gamma_alloc, beta_alloc, avg_mean_alloc, avg_var_alloc;

    private ScriptC_residualblock mResidualBlock;
    private ScriptC_batchnormalization rs_BN;
    private ScriptC_activation mActivation;
    private ScriptC_convolve2d mConvovle;


    public ResidualBlockChained(Context ctx, RenderScript rs, int in_channels, int out_channels, int ksize, int stride, int pad, int numBlocks) {
        super(ctx, rs);

        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.ksize = ksize;
        this.stride = stride;
        this.pad = pad;
        this.mNumBlocks = numBlocks;

        this.b = new float[out_channels];
        this.W = new float[out_channels * in_channels * ksize * ksize];

        // Pad the width of W to be multiple of 8.
        padded_Y_blas = in_channels * ksize * ksize;
        if (padded_Y_blas % 8 > 0) {
            padded_Y_blas = (padded_Y_blas / 8 + 1) * 8;
        }

        // Create Allocations for each convolution operation.
        W_alloc = new Allocation[mNumBlocks * 2];
        b_alloc = new Allocation[mNumBlocks * 2];
        Type.Builder tb = new Type.Builder(mRS, Element.F32(mRS));
        tb.setX(padded_Y_blas).setY(out_channels);
        for (int i = 0; i < mNumBlocks * 2; i++) {
            W_alloc[i] = Allocation.createTyped(mRS, tb.create());
        }
        Type.Builder tbeta = new Type.Builder(mRS, Element.F32(mRS));
        tbeta.setX(out_channels);
        for (int i = 0; i < mNumBlocks * 2; i++) {
            b_alloc[i] = Allocation.createTyped(mRS, tbeta.create());
        }


        // Create Allocations for each batch normalization operation.
        gamma = new float[out_channels];
        beta = new float[out_channels];
        avg_mean = new float[out_channels];
        avg_var = new float[out_channels];

        gamma_alloc = new Allocation[numBlocks * 2];
        beta_alloc = new Allocation[numBlocks * 2];
        avg_mean_alloc = new Allocation[numBlocks * 2];
        avg_var_alloc = new Allocation[numBlocks * 2];

        Type.Builder tbn = new Type.Builder(mRS, Element.F32(mRS));
        tbn.setX(out_channels);
        for (int i = 0; i < numBlocks * 2; i++) {
            gamma_alloc[i] = Allocation.createTyped(mRS, tbn.create());
            beta_alloc[i] = Allocation.createTyped(mRS, tbn.create());
            avg_mean_alloc[i] = Allocation.createTyped(mRS, tbn.create());
            avg_var_alloc[i] = Allocation.createTyped(mRS, tbn.create());
        }

        // Initialize the RS kernels;
        mResidualBlock = new ScriptC_residualblock(mRS);
        mActivation = new ScriptC_activation(mRS);
        mConvovle = new ScriptC_convolve2d(mRS);
        rs_BN = new ScriptC_batchnormalization(mRS);

        // Set the global variables for the convolution kernel.
        mConvovle.set_kernel_h(ksize);
        mConvovle.set_kernel_w(ksize);
        mConvovle.set_step_x(stride);
        mConvovle.set_step_y(stride);
        mConvovle.set_pad_h(pad);
        mConvovle.set_pad_w(pad);
        mConvovle.set_tile_h(TILE_Y);
    }

    // Load the data from file and transfer to corresponding Allocations.
    public void loadModel(String path) throws IOException {
        for (int i = 0; i < mNumBlocks; i++) {
            for (int j = 0; j < 2; j++) {
                // Read all convolution blocks.
                mInputStream = mContext.getAssets().open(path + "/r" + (i + 1) + "/c" + (j + 1) + "/W", AssetManager.ACCESS_BUFFER);
                ByteBuffer bb = readInput(mInputStream);
                FloatBuffer.wrap(W).put(bb.asFloatBuffer());

                // padding for GPU BLAS
                int W_height_input = in_channels * ksize * ksize;
                if (padded_Y_blas == W_height_input) {
                    // If the input width already satisfies the requirement, just copy to the Allocation.
                    W_alloc[i * 2 + j].copyFrom(W);
                } else {
                    // If not, a temp allocation needs to be created.
                    Allocation input = Allocation.createTyped(mRS,
                            Type.createXY(mRS, Element.F32(mRS), W_height_input, out_channels));
                    input.copyFrom(W);
                    W_alloc[i * 2 + j].copy2DRangeFrom(0, 0, W_height_input, out_channels, input, 0, 0);
                }

                mInputStream = mContext.getAssets().open(path + "/r" + (i + 1) + "/c" + (j + 1) + "/b", AssetManager.ACCESS_BUFFER);
                bb = readInput(mInputStream);
                FloatBuffer.wrap(b).put(bb.asFloatBuffer());
                b_alloc[i * 2 + j].copyFrom(b);

                // Read all batch normalization blocks;
                mInputStream = mContext.getAssets().open(path + "/r" + (i + 1) + "/b" + (j + 1) + "/gamma", AssetManager.ACCESS_BUFFER);
                bb = readInput(mInputStream);
                FloatBuffer.wrap(gamma).put(bb.asFloatBuffer());
                gamma_alloc[i * 2 + j].copyFrom(gamma);

                mInputStream = mContext.getAssets().open(path + "/r" + (i + 1) + "/b" + (j + 1) + "/beta", AssetManager.ACCESS_BUFFER);
                bb = readInput(mInputStream);
                FloatBuffer.wrap(beta).put(bb.asFloatBuffer());
                beta_alloc[i * 2 + j].copyFrom(beta);

                mInputStream = mContext.getAssets().open(path + "/r" + (i + 1) + "/b" + (j + 1) + "/avg_mean", AssetManager.ACCESS_BUFFER);
                bb = readInput(mInputStream);
                FloatBuffer.wrap(avg_mean).put(bb.asFloatBuffer());
                avg_mean_alloc[i * 2 + j].copyFrom(avg_mean);

                mInputStream = mContext.getAssets().open(path + "/r" + (i + 1) + "/b" + (j + 1) + "/avg_var", AssetManager.ACCESS_BUFFER);
                bb = readInput(mInputStream);
                FloatBuffer.wrap(avg_var).put(bb.asFloatBuffer());
                avg_var_alloc[i * 2 + j].copyFrom(avg_var);

            }

        }
        mInputStream.close();
        Log.v(TAG, "ResidualBlockChained loaded: " + b[0]);
    }


    public Allocation process(Allocation input, int img_h, int img_w) {
        // Set the input variables to the convolve kernel.
        mConvovle.set_img_h(img_h);
        mConvovle.set_img_w(img_w);
        mConvovle.set_img_channel(in_channels);

        // Set the input variables to batch normalization kernel.
        rs_BN.set_size(out_channels);

        // Calculate the dimensions of image after convolution.
        outH = ConvolveUtil.get_conv_outsize(img_h, ksize, stride, pad);
        outW = ConvolveUtil.get_conv_outsize(img_w, ksize, stride, pad);
        Log.v("ResidualBlock", "outH: " + outH + " outW: " + outW + " channels: " + in_channels + " " + out_channels);

        // Create the Allocations to hold the complete convolution results.
        Type.Builder tb = new Type.Builder(mRS, Element.F32(mRS));
        tb.setX(outH * outW).setY(out_channels);
        Allocation out_all = Allocation.createTyped(mRS, tb.create());
        Allocation in_all = Allocation.createTyped(mRS, tb.create());
        in_all.copyFrom(input);

        int padded_h = img_h + 2 * pad;
        int padded_w = img_w + 2 * pad;
        // Create Allocation to hold the padded image.
        Allocation img_padded = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), padded_h * padded_w, in_channels));
        mConvovle.forEach_zero(img_padded, img_padded);
        mConvovle.set_padded_alloc(img_padded);


        // Tiling in Y dimension
        int out_h_tile = ConvolveUtil.get_conv_outsize(TILE_Y, ksize, stride, pad);
        int out_w_tile = outW;
        Log.v(TAG, "tiled convolve size: " + out_h_tile + " " + out_w_tile);
        // Create the tiled column Allocation.
        Allocation col_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), out_h_tile * out_w_tile, padded_Y_blas));
        // Create the tiled output Allocation.
        Allocation out_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), out_h_tile * out_w_tile, out_channels));

        // Setup the parameters for paralleled im2col
        mConvovle.set_outH(out_h_tile);
        mConvovle.set_outW(out_w_tile);

        // The number of tiles, minimum 1.
        int nTiles = img_h / TILE_Y;
        if (nTiles == 0) nTiles = 1;

        // put all convolution and batch normalization in a loop.
        for (int ic = 0; ic < mNumBlocks; ic++) {
            long time;

            // 1st tiled convolution.
            mConvovle.set_img_alloc(in_all);
            mConvovle.invoke_padd();
            for (int it = 0; it < nTiles; it++) {
                mConvovle.set_tile_num(it);

                time = System.currentTimeMillis();
                mConvovle.forEach_im2col(col_alloc);

                if (LOG_TIME) {
                    mRS.finish();
                    time = System.currentTimeMillis() - time;
                    im2colTime += time;
                }

                time = System.currentTimeMillis();
                mBlas.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.NO_TRANSPOSE,
                        1.0f, W_alloc[ic * 2], col_alloc, 0.0f, out_alloc);
                if (LOG_TIME) {
                    mRS.finish();
                    time = System.currentTimeMillis() - time;
                    sgemmTime += time;
                }

                out_all.copy2DRangeFrom(it * out_h_tile * out_w_tile, 0, out_h_tile * out_w_tile, out_channels, out_alloc, 0, 0);
            }
            mConvovle.set_beta_alloc(b_alloc[ic * 2]);

            time = System.currentTimeMillis();
            mConvovle.forEach_addBeta(out_all, out_all);
            if (LOG_TIME) {
                mRS.finish();
                time = System.currentTimeMillis() - time;
                betaTime += time;
            }

            // 1st batch normalization
            rs_BN.set_beta_alloc(beta_alloc[ic * 2]);
            rs_BN.set_gamma_alloc(gamma_alloc[ic * 2]);
            rs_BN.set_mean_alloc(avg_mean_alloc[ic * 2]);
            rs_BN.set_var_alloc(avg_var_alloc[ic * 2]);

            time = System.currentTimeMillis();
            rs_BN.forEach_process(out_all, out_all);
            // 1st RELU
            mActivation.forEach_relu(out_all, out_all);
            if (LOG_TIME) {
                mRS.finish();
                time = System.currentTimeMillis() - time;
                normalizeTime += time;
            }

            // 2nd tiled convolution.
            mConvovle.set_img_alloc(out_all);
            mConvovle.invoke_padd();
            for (int it = 0; it < nTiles; it++) {
                mConvovle.set_tile_num(it);

                time = System.currentTimeMillis();
                mConvovle.forEach_im2col(col_alloc);

                if (LOG_TIME) {
                    mRS.finish();
                    time = System.currentTimeMillis() - time;
                    im2colTime += time;
                }

                time = System.currentTimeMillis();
                mBlas.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.NO_TRANSPOSE,
                        1.0f, W_alloc[ic * 2 + 1], col_alloc, 0.0f, out_alloc);
                if (LOG_TIME) {
                    mRS.finish();
                    time = System.currentTimeMillis() - time;
                    sgemmTime += time;
                }

                out_all.copy2DRangeFrom(it * out_h_tile * out_w_tile, 0, out_h_tile * out_w_tile, out_channels, out_alloc, 0, 0);
            }
            mConvovle.set_beta_alloc(b_alloc[ic * 2 + 1]);

            time = System.currentTimeMillis();
            mConvovle.forEach_addBeta(out_all, out_all);
            if (LOG_TIME) {
                mRS.finish();
                time = System.currentTimeMillis() - time;
                betaTime += time;
            }


            // 2nd batch normalization
            rs_BN.set_beta_alloc(beta_alloc[ic * 2 + 1]);
            rs_BN.set_gamma_alloc(gamma_alloc[ic * 2 + 1]);
            rs_BN.set_mean_alloc(avg_mean_alloc[ic * 2 + 1]);
            rs_BN.set_var_alloc(avg_var_alloc[ic * 2 + 1]);

            time = System.currentTimeMillis();
            rs_BN.forEach_process(out_all, out_all);
            if (LOG_TIME) {
                mRS.finish();
                time = System.currentTimeMillis() - time;
                normalizeTime += time;
            }

            // Add the residual with the input.
            mResidualBlock.set_img_alloc(in_all);
            mResidualBlock.forEach_add(out_all, out_all);

            Allocation temp = in_all;
            in_all = out_all;
            out_all = temp;
        }

        // Destroy the intermediate Allocations.
        img_padded.destroy();
        col_alloc.destroy();
        out_alloc.destroy();

        return in_all;
    }
}
