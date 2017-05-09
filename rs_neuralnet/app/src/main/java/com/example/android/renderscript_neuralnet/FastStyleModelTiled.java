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
import android.graphics.Bitmap;
import android.os.Build;
import android.support.v8.renderscript.Allocation;
import android.support.v8.renderscript.Element;
import android.support.v8.renderscript.RenderScript;
import android.support.v8.renderscript.ScriptIntrinsicBLAS;
import android.support.v8.renderscript.ScriptIntrinsicBlur;
import android.support.v8.renderscript.ScriptIntrinsicConvolve3x3;
import android.support.v8.renderscript.Type;
import android.util.Log;

import java.io.IOException;

/**
 * Created by miaowang on 8/25/16.
 */

/*
   FastStyle Convolutional Neural Net model.
   The structure of the neural net is like the following, from top to bottom:

                   [2D Convolution Layer]
                             |
                   [Batch Normalization]
                             |
                   [2D Convolution Layer]
                             |
                   [Batch Normalization]
                             |
                   [2D Convolution Layer]
                             |
                   [Batch Normalization]
                             |
                      [Residual Block]-------|
                             |               |
                      [Residual Block]       |
                             |               |
                      [Residual Block]       |-> Could be chained together
                             |               |
                      [Residual Block]       |
                             |               |
                      [Residual Block]-------|
                             |
                   [2D Deconvolution Layer]
                             |
                   [Batch Normalization]
                             |
                   [2D Deconvolution Layer]
                             |
                   [Batch Normalization]
                             |
                   [2D Deconvolution Layer]


   Unlike FastStyleModel, in this special implementation (FastStyleModelTiled), all the convolution
   and deconvolution layers are tiled to reduce the memory pressure. Furthermore, Residual Blocks
   are chained together so that the temporary Allocations created can be reused, further decrease
   the memory footprint and improve the overall performance.
*/
public class FastStyleModelTiled {
    public String mModel = null;
    private boolean mLoaded = false;
    private boolean LOG_TIME = NeuralNetLayerBase.LOG_TIME;
    
    private static final String DEFAULT_MODEL = "composition";
    private static final String TAG = "FloatFastStyleModel";
    
    static int MAX_IMG_SIZE = 256;
    static int MAX_CHUNK_SIZE = 256;

    private Context mContext;
    private Convolution2DTiled[] mConvLayer;
    private ResidualBlockChained mResidualLayer;
    private Deconvolution2DTiled[] mDeconvLayer;
    private BatchNormalization[] mBatchNormLayer;
    
    private RenderScript mRS;
    private ScriptIntrinsicBLAS mBlas;
    private ScriptC_img2alloc mImg2Alloc;
    private ScriptC_activation mActivation;


    public FastStyleModelTiled(Context ctx) {
        mContext = ctx;

        mRS = RenderScript.create(ctx, Build.VERSION_CODES.LOLLIPOP);
        mBlas = ScriptIntrinsicBLAS.create(mRS);
        mImg2Alloc = new ScriptC_img2alloc(mRS);
        mActivation = new ScriptC_activation(mRS);

        mConvLayer = new Convolution2DTiled[3];
        mResidualLayer = new ResidualBlockChained(ctx, mRS, 128, 128, 3, 1, 1, 5);
        mDeconvLayer = new Deconvolution2DTiled[3];
        mBatchNormLayer = new BatchNormalization[5];

        mConvLayer[0] = new Convolution2DTiled(ctx, mRS, 3, 32, 9, 1, 4);
        mConvLayer[1] = new Convolution2DTiled(ctx, mRS, 32, 64, 4, 2, 1);
        mConvLayer[2] = new Convolution2DTiled(ctx, mRS, 64, 128, 4, 2, 1);

        mDeconvLayer[0] = new Deconvolution2DTiled(ctx, mRS, 128, 64, 4, 2, 1);
        mDeconvLayer[1] = new Deconvolution2DTiled(ctx, mRS, 64, 32, 4, 2, 1);
        mDeconvLayer[2] = new Deconvolution2DTiled(ctx, mRS, 32, 3, 9, 1, 4);

        mBatchNormLayer[0] = new BatchNormalization(ctx, mRS, 32);
        mBatchNormLayer[1] = new BatchNormalization(ctx, mRS, 64);
        mBatchNormLayer[2] = new BatchNormalization(ctx, mRS, 128);
        mBatchNormLayer[3] = new BatchNormalization(ctx, mRS, 64);
        mBatchNormLayer[4] = new BatchNormalization(ctx, mRS, 32);
    }

    public void loadModel() throws IOException {
        loadModel(DEFAULT_MODEL);
    }

    // Load data to each layer.
    public void loadModel(String modelName) throws IOException {
        if (modelName == null) {
            modelName = DEFAULT_MODEL;
        }
        for (int i = 1; i <= mConvLayer.length; i++) {
            mConvLayer[i - 1].loadModel(modelName + "/c" + i);
        }

        mResidualLayer.loadModel(modelName);

        for (int i = 1; i <= mDeconvLayer.length; i++) {
            mDeconvLayer[i - 1].loadModel(modelName + "/d" + i);
        }
        for (int i = 1; i <= mBatchNormLayer.length; i++) {
            mBatchNormLayer[i - 1].loadModel(modelName + "/b" + i);
        }
        mLoaded = true;
    }

    private Allocation processImgChunk(Bitmap bitmap) {
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();

        mImg2Alloc.set_height(height);
        mImg2Alloc.set_weight(width);

        Bitmap outImg = Bitmap.createBitmap(bitmap);
        // RGB bitmap Allocation.
        Allocation imgAlloc = Allocation.createFromBitmap(mRS, bitmap);
        mImg2Alloc.set_img_alloc(imgAlloc);
        // Float input Allocation.
        Allocation result = Allocation.createTyped(mRS, Type.createXY(mRS, Element.F32(mRS), height * width, 3));
        // convert the bitmap to 3 * (h * w) float Allocation;
        mImg2Alloc.forEach_img2alloc(result);

        // Actual computation;
        // 1st Convolution layer.
        result = mConvLayer[0].process(result, height, width);
        // Use ELU for activation.
        mActivation.forEach_elu(result, result);
        // 1st Batch Normalization.
        mBatchNormLayer[0].process(result);

        // 2nd Convolution layer.
        result = mConvLayer[1].process(result, mConvLayer[0].outH, mConvLayer[0].outW);
        mActivation.forEach_elu(result, result);
        // 2nd Batch Normalization.
        mBatchNormLayer[1].process(result);

        // 3rd Convolution layer.
        result = mConvLayer[2].process(result, mConvLayer[1].outH, mConvLayer[1].outW);
        mActivation.forEach_elu(result, result);
        // 3rd Batch Normalization.
        mBatchNormLayer[2].process(result);

        // Process through the entire residual block.
        result = mResidualLayer.process(result, mConvLayer[2].outH, mConvLayer[2].outW);

        // 1st Deconvolution layer.
        result = mDeconvLayer[0].process(result, mResidualLayer.outH, mResidualLayer.outW);
        mActivation.forEach_elu(result, result);
        // 4th Batch Normalization.
        mBatchNormLayer[3].process(result);

        // 2nd Deconvolution layer.
        result = mDeconvLayer[1].process(result, mDeconvLayer[0].outH, mDeconvLayer[0].outW);
        mActivation.forEach_elu(result, result);
        // 5th Batch Normalization.
        mBatchNormLayer[4].process(result);

        // 3rd Deconvolution layer.
        result = mDeconvLayer[2].process(result, mDeconvLayer[1].outH, mDeconvLayer[1].outW);

        // Convert floating point result to RGB image.
        mImg2Alloc.set_nn_alloc(result);
        Allocation outAlloc = Allocation.createFromBitmap(mRS, outImg);
        mImg2Alloc.forEach_alloc2img(outAlloc);
        return outAlloc;
    }

    public Bitmap processImage(Bitmap bitmap) {
        ScriptC_network script = new ScriptC_network(mRS);

        int numElements = 100;
        Element floatElement = Element.I32(mRS);
        Type arrayType = Type.createX(mRS, floatElement,  numElements);
        Allocation inputAlloc = Allocation.createTyped(mRS, arrayType);
        Allocation outputAlloc = Allocation.createTyped(mRS, arrayType);

        script.forEach_mapper(inputAlloc, outputAlloc);

        int[] output = new int[numElements];
        outputAlloc.copyTo(output);
        Log.i(TAG, output[0] + " " + output[99]);


        if (!mLoaded) {
            try {
                loadModel();
            } catch (IOException e) {

            }
        }
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();

        // Crop the image.
        Bitmap outImgBig = Bitmap.createBitmap(bitmap, (width - MAX_IMG_SIZE) / 2,
                (height - MAX_IMG_SIZE) / 2, MAX_IMG_SIZE, MAX_IMG_SIZE);
        // Process the cropped image through the neural net.
        Allocation outImgBigAlloc = processImgChunk(outImgBig);

        // Blur the output image a bit.
        Allocation blurredAlloc = Allocation.createFromBitmap(mRS, outImgBig);
        ScriptIntrinsicBlur mBlur = ScriptIntrinsicBlur.create(mRS, Element.U8_4(mRS));
        mBlur.setInput(outImgBigAlloc);
        mBlur.setRadius(1.5f);
        mBlur.forEach(blurredAlloc);
        blurredAlloc.copyTo(outImgBig);

        // outImgBigAlloc.copyTo(outImgBig);
        ScriptIntrinsicConvolve3x3 convolution = ScriptIntrinsicConvolve3x3.create(mRS, Element.U8_4(mRS));
        float[] matrix_sharpen =
                        { 0, -1, 0,
                         -1, 5, -1,
                          0, -1, 0};
        convolution.setInput(blurredAlloc);
        convolution.setCoefficients(matrix_sharpen);
        convolution.forEach(outImgBigAlloc);
        outImgBigAlloc.copyTo(outImgBig);

        logBenchmarkResult();
        return outImgBig;
    }

    public void logBenchmarkResult() {
        if (LOG_TIME) {
            BenchmarkResult result = new BenchmarkResult();
            for (Convolution2DTiled aMConvLayer : mConvLayer) {
                aMConvLayer.getBenchmark(result);
            }
            for (Deconvolution2DTiled aMDeconvLayer : mDeconvLayer) {
                aMDeconvLayer.getBenchmark(result);
            }
            mResidualLayer.getBenchmark(result);
            for (BatchNormalization aMBatchNormLayer : mBatchNormLayer) {
                aMBatchNormLayer.getBenchmark(result);
            }
            Log.v(TAG, "SGEMM Time: " + result.sgemmTime + ", im2col Time: " + result.im2colTime +
                    ", col2im Time: " + result.col2imTime + ", beta Time: " + result.betaTime +
                    ", normalize Time: " + result.normalizeTime);
        }
    }
}
