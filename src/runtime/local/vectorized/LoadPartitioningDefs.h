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

enum QueueTypeOption {
    CENTRALIZED=0,
    PERGROUP,
    PERCPU
};

enum VictimSelectionLogic {
    SEQ=0,
    SEQPRI,
    RANDOM,
    RANDOMPRI
};

enum SelfSchedulingScheme {
    STATIC=0,
    SS,
    GSS,
    TSS,
    FAC2,
    TFSS,
    FISS,
    VISS,
    PLS,
    MSTATIC,
    MFSC,
    PSS,
    AUTO,
    INVALID=-1 /* only for JSON enum conversion */
};
