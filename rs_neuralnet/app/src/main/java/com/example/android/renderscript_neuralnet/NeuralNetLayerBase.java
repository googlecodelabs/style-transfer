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
import android.support.v8.renderscript.RenderScript;
import android.support.v8.renderscript.ScriptIntrinsicBLAS;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by miaowang on 8/15/16.
 */
public abstract class NeuralNetLayerBase {
    public static final String TAG = "FastStyleModel";
    public static final boolean LOG_TIME = true;

    public InputStream mInputStream;
    public Context mContext;
    public RenderScript mRS;
    public ScriptIntrinsicBLAS mBlas;

    public long sgemmTime = 0;
    public long normalizeTime = 0;
    public long im2colTime = 0;
    public long col2imTime = 0;
    public long betaTime = 0;
    public long conv2dTime = 0;

    public NeuralNetLayerBase(Context ctx, RenderScript rs) {
        mContext = ctx;
        mRS = rs;
        mBlas = ScriptIntrinsicBLAS.create(mRS);
    }

    abstract public void loadModel(String path) throws IOException;

    public ByteBuffer readInput(InputStream inputStream) throws IOException {
        // this dynamically extends to take the bytes you read
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

        // this is storage overwritten on each iteration with bytes
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        // we need to know how may bytes were read to write them to the byteBuffer
        int len = 0;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }

        // and then we can return your byte array.
        return ByteBuffer.wrap(byteBuffer.toByteArray()).order(ByteOrder.nativeOrder());
    }

    public void getBenchmark(BenchmarkResult result) {
        result.sgemmTime += sgemmTime;
        result.normalizeTime += normalizeTime;
        result.im2colTime += im2colTime;
        result.col2imTime += col2imTime;
        result.betaTime += betaTime;
        result.conv2dTime += conv2dTime;

        sgemmTime = 0;
        normalizeTime = 0;
        im2colTime = 0;
        col2imTime = 0;
        betaTime = 0;
        conv2dTime = 0;
    }
}
