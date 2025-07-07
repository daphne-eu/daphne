/*
 * Copyright 2025 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <runtime/local/context/DaphneContext.h>

#include <string>

/**
 * @brief Executes instrumentation code before a kernel is called.
 * Currently only starts the statistics runtime tracking when --statistics is
 * specified by the user.
 */
void preKernelInstrumentation(int kId, DaphneContext *ctx);

/**
 * @brief Executes instrumentation code after a kernel call returned.
 * Currently only stops the statistics runtime tracking when --statistics is
 * specified by the user.
 */
void postKernelInstrumentation(int kId, DaphneContext *ctx);
