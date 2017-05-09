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

int height, weight;
int padding;

// Input RGB Allocation.
rs_allocation img_alloc;
// Convert RGB image to float Allocation.
float RS_KERNEL img2alloc(uint32_t x, uint32_t y) {
    int imgY = x / weight;
    int imgX = x - imgY * weight;
    uchar4 rgb = rsGetElementAt_uchar4(img_alloc, imgX, imgY);
    return (float)rgb[y];
}

// Output Allocation of neural net.
rs_allocation nn_alloc;
// Convert the float neural net output to RGB image.
uchar4 RS_KERNEL alloc2img(uint32_t x, uint32_t y) {
    int nnX = y * weight + x;
    uchar4 out;
    out.r = (tanh(rsGetElementAt_float(nn_alloc, nnX, 0)) + 1) * 127.5;
    out.g = (tanh(rsGetElementAt_float(nn_alloc, nnX, 1)) + 1) * 127.5;
    out.b = (tanh(rsGetElementAt_float(nn_alloc, nnX, 2)) + 1) * 127.5;
    return out;
}