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
    Two-dimensional tiled deconvolution (transposed convolution) layer.

    Attributes:
    in_channels  :  Number of channels of input img.
    out_channels :  Number of channels of output img.
    ksize        :  Size of filters / kernels.
    stride       :  Stride of filters / kernels.
    pad          :  Spatial padding width for input img.
    W            :  Weight parameter.
    b            :  Bias parameter.
*/
public class Deconvolution2DTiled extends NeuralNetLayerBase {
    // The dimension in Y for each tile.
    private final int TILE_Y = 64;

    // The dimension of the image after convolution.
    // Used by subsequent operations (layers).
    public int outH, outW;

    private int in_channels, out_channels;
    private int ksize, stride, pad;
    private float[] W;
    private float[] b;

    // The padded dimension to satisfy alignment requirement for certain GPUs.
    private int padded_Y_blas;

    private ScriptC_deconvolve2d mConvovle;
    private Allocation W_alloc, b_alloc;

    public Deconvolution2DTiled(Context ctx, RenderScript rs, int in_channels, int out_channels, int ksize, int stride, int pad) {
        super(ctx, rs);

        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.ksize = ksize;
        this.stride = stride;
        this.pad = pad;
        // X dimension for W: in_channels * ksize * ksize
        // Y dimension for W: out_channels
        this.W = new float[out_channels * ksize * ksize * in_channels];
        this.b = new float[out_channels];

        // Pad the width of W to be multiple of 8.
        padded_Y_blas = out_channels * ksize * ksize;
        if (padded_Y_blas % 8 > 0) {
            padded_Y_blas = (padded_Y_blas / 8 + 1) * 8;
        }

        // Create Allocations for W and b.
        W_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), in_channels, padded_Y_blas));
        b_alloc = Allocation.createSized(mRS, Element.F32(mRS), out_channels);

        // Initialize the 2D deconvolution kernel;
        mConvovle = new ScriptC_deconvolve2d(mRS);

        // Set the global variables for the RS kernel.
        mConvovle.set_tile_h(TILE_Y);
        mConvovle.set_col_h(TILE_Y);

        mConvovle.set_kernel_h(ksize);
        mConvovle.set_kernel_w(ksize);
        mConvovle.set_step_x(stride);
        mConvovle.set_step_y(stride);
        mConvovle.set_pad_h(pad);
        mConvovle.set_pad_w(pad);
        mConvovle.set_beta_alloc(b_alloc);
    }

    // Load the data from file and transfer to corresponding Allocations.
    public void loadModel(String path) throws IOException {
        mInputStream = mContext.getAssets().open(path + "/W", AssetManager.ACCESS_BUFFER);
        ByteBuffer bb = readInput(mInputStream);
        FloatBuffer.wrap(W).put(bb.asFloatBuffer());

        // Tranpose W after loading the data.
        float[] w_trans = new float[in_channels * padded_Y_blas];
        for (int i = 0; i < out_channels * ksize * ksize; i++) {
            for (int j = 0; j < in_channels; j++) {
                w_trans[i * in_channels + j] = W[j * out_channels * ksize * ksize + i];
            }
        }
        W_alloc.copyFrom(w_trans);

        mInputStream = mContext.getAssets().open(path + "/b", AssetManager.ACCESS_BUFFER);
        bb = readInput(mInputStream);
        FloatBuffer.wrap(b).put(bb.asFloatBuffer());
        b_alloc.copyFrom(b);

        mInputStream.close();
        Log.v(TAG, "Deconvolution2D loaded: " + b[0]);
    }


    /*
        The workflow of 2D deconvolution:
        1. Use matrix multiplication API to calculate the tiled deconvolution.
        2. Rearrange the tiled column image by col2im.
        3. Repeat 1~2 until the entire image is traversed.
        4. Unpad the output image.
     */
    public Allocation process(Allocation input, int col_h, int col_w) {
        // Calculate the dimensions of image after deconvolution.
        outH = ConvolveUtil.get_deconv_outsize(col_h, ksize, stride, pad);
        outW = ConvolveUtil.get_deconv_outsize(col_w, ksize, stride, pad);

        // Set the global variables for the RS kernel.
        mConvovle.set_col_w(col_w);
        mConvovle.set_col_channel(out_channels);
        mConvovle.set_img_channel(out_channels);
        mConvovle.set_img_h(outH);
        mConvovle.set_img_w(outW);


        int tiledDimX = TILE_Y * col_w;
        int tiledDimY = in_channels;
        // Create the tiled input Allocation.
        Allocation tiledIn_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), tiledDimX, tiledDimY));

        // Create the tiled output Allocation.
        Allocation tiledOut_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), tiledDimX, padded_Y_blas));
        mConvovle.set_col_alloc(tiledOut_alloc);

        // Create Allocation to hold the padded image.
        int padded_h = outH + 2 * pad;
        int padded_w = outW + 2 * pad;
        Allocation img_padded = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), padded_h * padded_w, out_channels));
        // Initialize the padded Allocation to zero.
        mConvovle.forEach_zero(img_padded, img_padded);
        mConvovle.set_padded_alloc(img_padded);

        // Create final output image Allocation
        Allocation img_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), outH * outW, out_channels));
        mConvovle.set_img_alloc(img_alloc);

        // The number of tiles, minimum 1.
        int nTiles = col_h / TILE_Y;
        if (nTiles == 0) nTiles = 1;

        long time;

        // Iterate each tile for 2D deconvolution and copy to the final output.
        for (int it = 0; it < nTiles; it++) {
            // Set the current tile number;
            mConvovle.set_tile_num(it);

            // Copy data to the tiled input Allocation.
            tiledIn_alloc.copy2DRangeFrom(0, 0, tiledDimX, in_channels, input, it * tiledDimX, 0);
            time = System.currentTimeMillis();

            // Conduct the deconvolution by matrix multiplication, using SGEMM (BLAS API).
            mBlas.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.NO_TRANSPOSE,
                    1.0f, W_alloc, tiledIn_alloc, 0.0f, tiledOut_alloc);
            if (LOG_TIME) {
                mRS.finish();
                time = System.currentTimeMillis() - time;
                sgemmTime += time;
            }

            time = System.currentTimeMillis();
            // Invoke col2im kernel, to transform column image to padded image:
            mConvovle.invoke_col2im_tileY();
            if (LOG_TIME) {
                mRS.finish();
                time = System.currentTimeMillis() - time;
                col2imTime += time;
            }

        }

        // Invoked the unpadding kernel.
        mConvovle.invoke_unpadd();

        time = System.currentTimeMillis();
        // Add beta to the results for each channel.
        mConvovle.forEach_addBeta(img_alloc, img_alloc);
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            betaTime += time;
        }

        // Destroy the intermediate Allocations.
        tiledOut_alloc.destroy();
        tiledIn_alloc.destroy();
        img_padded.destroy();

        return img_alloc;
    }
}
