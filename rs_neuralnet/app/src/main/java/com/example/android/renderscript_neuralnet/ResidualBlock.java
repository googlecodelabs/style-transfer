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
import android.support.v8.renderscript.Allocation;
import android.support.v8.renderscript.RenderScript;

import java.io.IOException;

/**
 * Created by miaowang on 8/15/16.
 */

/*
    Reference implementation of 2D Residual Block layer.
    Each Residual Block consists of 2 Convolution layers and two BatchNormalization layers.
    The calculated residual will be added with the input image.

    Attributes:
    n_in  :  Number of channels of input arrays.
    n_out :  Number of channels of output arrays.
*/
public class ResidualBlock extends NeuralNetLayerBase {
    // The dimension of the image after ResidualBlock.
    // Used by subsequent operations (layers).
    public int outH, outW;

    private int n_in, n_out;
    private int stride = 1;
    private int ksize = 3;
    private Convolution2D c1;
    private Convolution2D c2;
    private BatchNormalization b1;
    private BatchNormalization b2;

    private ScriptC_residualblock mResidualBlock;
    private ScriptC_activation mActivation;

    public ResidualBlock(Context ctx, RenderScript rs, int n_in, int n_out) {
        super(ctx, rs);
        this.n_in = n_in;
        this.n_out = n_out;

        double w = Math.sqrt(2);
        c1 = new Convolution2D(ctx, rs, n_in, n_out, ksize, stride, 1);
        c2 = new Convolution2D(ctx, rs, n_out, n_out, ksize, stride, 1);
        b1 = new BatchNormalization(ctx, rs, n_out);
        b2 = new BatchNormalization(ctx, rs, n_out);

        // Initialize the RS kernels;
        mResidualBlock = new ScriptC_residualblock(mRS);
        mActivation = new ScriptC_activation(mRS);
    }

    // Load the data for each sub-layer.
    public void loadModel(String path) throws IOException {
        c1.loadModel(path + "/c1");
        c2.loadModel(path + "/c2");
        b1.loadModel(path + "/b1");
        b2.loadModel(path + "/b2");
    }

    public void getBenchmark(BenchmarkResult result) {
        c1.getBenchmark(result);
        c2.getBenchmark(result);
        b1.getBenchmark(result);
        b2.getBenchmark(result);
    }

    public Allocation process(Allocation input, int height, int width) {
        // 1st convolution.
        Allocation output = c1.process(input, height, width);
        // 1st batch normalization.
        b1.process(output);
        // Use RELU for the activation function.
        mActivation.forEach_relu(output, output);
        // 2nd convolution.
        output = c2.process(output, c1.outH, c1.outW);
        // 2nd batch normalization.
        b2.process(output);

        // Add the residual back to the input image.
        mResidualBlock.set_img_alloc(input);
        mResidualBlock.forEach_add(output, output);

        // Update the output dimensions.
        outH = c2.outH;
        outW = c2.outW;
        return output;
    }
}
