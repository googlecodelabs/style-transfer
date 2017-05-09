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

rs_allocation mean_alloc, var_alloc, gamma_alloc, beta_alloc;
int size;

// Batch normalization kernel.
float RS_KERNEL process(float in, uint32_t x, uint32_t y) {
   int pos = y;

   float mean = rsGetElementAt_float(mean_alloc, pos);
   float gamma = rsGetElementAt_float(gamma_alloc, pos);
   float beta = rsGetElementAt_float(beta_alloc, pos);
   float std = half_sqrt(rsGetElementAt_float(var_alloc, pos));

   float out = (in - mean) / std * gamma + beta;
   return out;
}