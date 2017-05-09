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
    Two-dimensional Reference implementation of 2D convolutional layer.

    Attributes:
    in_channels  :  Number of channels of input img.
    out_channels :  Number of channels of output img.
    ksize        :  Size of filters / kernels.
    stride       :  Stride of filters / kernels.
    pad          :  Spatial padding width for input img.
    W            :  Weight parameter.
    b            :  Bias parameter.
*/
public class Convolution2D extends NeuralNetLayerBase {
    // The dimension of the image after convolution.
    // Used by subsequent operations (layers).
    public int outH, outW;

    private int in_channels, out_channels;
    private int ksize, stride, pad;
    private float[] W;
    private float[] b;

    // The padded dimension to satisfy alignment requirement for certain GPUs.
    private int padded_Y_blas;

    private ScriptC_convolve2d mConvovle;
    private Allocation W_alloc, b_alloc;

    public Convolution2D(Context ctx, RenderScript rs, int in_channels, int out_channels, int ksize, int stride, int pad) {
        super(ctx, rs);

        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.ksize = ksize;
        this.stride = stride;
        this.pad = pad;
        // X dimension for W: in_channels * ksize * ksize
        // Y dimension for W: out_channels
        this.W = new float[out_channels * in_channels * ksize * ksize];
        this.b = new float[out_channels];

        // Pad the width of W to be multiple of 8.
        padded_Y_blas = in_channels * ksize * ksize;
        if (padded_Y_blas % 8 > 0) {
            padded_Y_blas = (padded_Y_blas / 8 + 1) * 8;
        }

        // Create Allocations for W and b.
        W_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), padded_Y_blas, out_channels));
        b_alloc = Allocation.createSized(mRS, Element.F32(mRS), out_channels);

        // Initialize the 2D convolution kernel;
        mConvovle = new ScriptC_convolve2d(mRS);

        // Set the global variables for the RS kernel.
        mConvovle.set_kernel_h(ksize);
        mConvovle.set_kernel_w(ksize);
        mConvovle.set_step_x(stride);
        mConvovle.set_step_y(stride);
        mConvovle.set_pad_h(pad);
        mConvovle.set_pad_w(pad);

        mConvovle.set_beta_alloc(b_alloc);
        mConvovle.set_img_channel(in_channels);
    }

    // Load the data from file and transfer to corresponding Allocations.
    public void loadModel(String path) throws IOException {
        mInputStream = mContext.getAssets().open(path + "/W", AssetManager.ACCESS_BUFFER);
        ByteBuffer bb = readInput(mInputStream);
        FloatBuffer.wrap(W).put(bb.asFloatBuffer());

        // padding for GPU BLAS when necessary.
        int W_height_input = in_channels * ksize * ksize;
        if (padded_Y_blas == W_height_input) {
            // If the input width already satisfies the requirement, just copy to the Allocation.
            W_alloc.copyFrom(W);
        } else {
            // If not, a temp allocation needs to be created.
            Allocation input = Allocation.createTyped(mRS,
                    Type.createXY(mRS, Element.F32(mRS), W_height_input, out_channels));
            input.copyFrom(W);
            W_alloc.copy2DRangeFrom(0, 0, W_height_input, out_channels, input, 0, 0);
        }

        mInputStream = mContext.getAssets().open(path + "/b", AssetManager.ACCESS_BUFFER);
        bb = readInput(mInputStream);
        FloatBuffer.wrap(b).put(bb.asFloatBuffer());
        b_alloc.copyFrom(b);

        mInputStream.close();
        Log.v(TAG, "Convolution2D loaded: " + b[0]);
    }


    /*
        The workflow of 2D convolution:
        1. Pad the input image
        2. Rearrange the image by im2col
        3. Use matrix multiplication API to calculate the convolution.
     */
    public Allocation process(Allocation input, int img_h, int img_w) {
        // Set the input variables to the convolve kernel.
        mConvovle.set_img_h(img_h);
        mConvovle.set_img_w(img_w);
        mConvovle.set_img_alloc(input);

        // Calculate the dimensions of the image after padding.
        int padded_h = img_h + 2 * pad;
        int padded_w = img_w + 2 * pad;

        // Create Allocation to hold the padded image.
        Allocation img_padded = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), padded_h * padded_w, in_channels));
        // Initialize the padded Allocation to zero.
        mConvovle.forEach_zero(img_padded, img_padded);
        mConvovle.set_padded_alloc(img_padded);

        // Invoked the padding kernel.
        mConvovle.invoke_padd();

        // TODO: Use convolve2DGEMM instead.
        Allocation out_alloc = convolve2D(img_padded, img_h, img_w);

        // Destroy the intermediate Allocations.
        img_padded.destroy();

        return out_alloc;
    }

    private Allocation convolve2DGEMM(Allocation img_padded, int img_h, int img_w) {
        // Calculate the dimensions of image after convolution.
        int out_h = ConvolveUtil.get_conv_outsize(img_h, ksize, stride, pad);
        int out_w = ConvolveUtil.get_conv_outsize(img_w, ksize, stride, pad);
        Log.v(TAG, "convolve size: " + out_h + " " + out_w);
        // Create the column Allocation.
        Allocation col_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), out_h * out_w, padded_Y_blas));

        long time = System.currentTimeMillis();
        // Invoke im2col kernel, to transform padded image to column image:
        mConvovle.set_outW(out_w);
        mConvovle.set_outH(out_h);
        mConvovle.forEach_im2col(col_alloc);

        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            im2colTime += time;
            Log.v(TAG, "Convolution2D, channels: " + in_channels + ", " + out_channels + " size: " + img_h + ", " + img_w + " im2col process time: " + time);
        }

        // Create the output Allocation for SGEMM operation.
        Allocation out_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), out_h * out_w, out_channels));

        time = System.currentTimeMillis();
        // Conduct the convolution by matrix multiplication, using SGEMM (BLAS API).
        mBlas.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.NO_TRANSPOSE,
                1.0f, W_alloc, col_alloc, 0.0f, out_alloc);

        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            sgemmTime += time;
            Log.v(TAG, "Convolution2D, channels: " + in_channels + ", " + out_channels + " size: " + img_h + ", " + img_w + " SGEMM process time: " + time);
        }

        time = System.currentTimeMillis();
        // Add beta to the results for each channel.
        mConvovle.forEach_addBeta(out_alloc, out_alloc);
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            betaTime += time;
            Log.v(TAG, "Convolution2D, channels: " + in_channels + ", " + out_channels + " size: " + img_h + ", " + img_w + " initBeta process time: " + time);
        }

        // Destroy the intermediate Allocations.
        col_alloc.destroy();

        // Update the output dimensions.
        outH = out_h;
        outW = out_w;

        return out_alloc;
    }

    private Allocation convolve2D(Allocation img_padded, int img_h, int img_w) {
        // Calculate the dimensions of image after convolution.
        int out_h = ConvolveUtil.get_conv_outsize(img_h, ksize, stride, pad);
        int out_w = ConvolveUtil.get_conv_outsize(img_w, ksize, stride, pad);
        Log.v(TAG, "convolve size: " + out_h + " " + out_w);

        mConvovle.set_outW(out_w);
        mConvovle.set_outH(out_h);
        mConvovle.set_W_alloc(W_alloc);
        mConvovle.set_input_padded(img_padded);

        // Create the output Allocation for 2D convolution operation.
        Allocation out_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), out_h * out_w, out_channels));

        long time = System.currentTimeMillis();
        mConvovle.forEach_convolve2D(out_alloc);
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            conv2dTime += time;
            Log.v(TAG, "Convolution2D, channels: " + in_channels + ", " + out_channels + " size: " + img_h + ", " + img_w + " conv2D process time: " + time);
        }

        time = System.currentTimeMillis();
        mConvovle.forEach_addBeta(out_alloc, out_alloc);
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            betaTime += time;
            Log.v(TAG, "Convolution2D, channels: " + in_channels + ", " + out_channels + " size: " + img_h + ", " + img_w + " initBeta process time: " + time);
        }
        // Update the output dimensions.
        outH = out_h;
        outW = out_w;

        return out_alloc;
    }
}