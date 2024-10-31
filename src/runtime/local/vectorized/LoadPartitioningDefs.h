/*
 * Copyright 2021 The DAPHNE Consortium
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

#pragma once

enum class QueueTypeOption { CENTRALIZED, PERGROUP, PERCPU };

enum class VictimSelectionLogic { SEQ, SEQPRI, RANDOM, RANDOMPRI };

enum class SelfSchedulingScheme {
    INVALID = -1,
    STATIC,
    SS,   // self-scheduling
    GSS,  // guided self-scheduling
    TSS,  // trapezoid self-scheduling
    FAC2, // factoring
    TFSS, // trapezoid factoring self-scheduling (TFSS)
    FISS, // fixed increase self-scheduling
    VISS, // variable increase self-scheduling
    PLS,  // performance-based loop self-scheduling
    PSS,  // probabilistic self-scheduling
    MSTATIC,
    MFSC, // modifed fixed-size chunk self-scheduling
    AUTO,
};
