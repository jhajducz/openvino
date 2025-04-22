// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(1, WG_SIZE, 1)))
REQD_SUB_GROUP_SIZE(SG_SIZE)
#endif
KERNEL(mvn_byxf_within_channels_sg_over_x)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(2);
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_spatial_start = get_sub_group_id() * SG_SIZE * TILE_SIZE;

    const uint x_start = (sg_spatial_start % (TILE_SIZE_X * SG_SIZE)) + sg_local_id * TILE_SIZE_X;
    const uint y_start = (sg_spatial_start / (TILE_SIZE_X * SG_SIZE));

    ACCUMULATOR_TYPE in_vals_mean = (ACCUMULATOR_TYPE)0;
#if NORMALIZE_VARIANCE == 1
    ACCUMULATOR_TYPE in_vals_squared_mean = (ACCUMULATOR_TYPE)0;
    ACCUMULATOR_TYPE inv_sqrt_var = (ACCUMULATOR_TYPE)0;
#endif

    ACCUMULATOR_TYPE in_vals[TILE_SIZE_Y][TILE_SIZE_X];

    uint idx = GET_DATA_INDEX(INPUT0, b, f, y_start, x_start);
    __attribute__((opencl_unroll_hint))
    for (int y = 0; y < TILE_SIZE_Y; y++) {
        __attribute__((opencl_unroll_hint))
        for (int x = 0; x < TILE_SIZE_X; x++) {
            in_vals[y][x] = (ACCUMULATOR_TYPE)input[idx];
            in_vals_mean += sub_group_reduce_add(in_vals[y][x]);
#if NORMALIZE_VARIANCE == 1
            in_vals_squared_mean += sub_group_reduce_add(in_vals[y][x] * in_vals[y][x]);
#endif
            idx += INPUT0_X_PITCH;
        }
        idx += INPUT0_Y_PITCH - TILE_SIZE_X * INPUT0_X_PITCH;
    }
    idx -= TILE_SIZE_Y * INPUT0_Y_PITCH;

    in_vals_mean = work_group_reduce_add(in_vals_mean) / (SG_SIZE * DATA_SET_SIZE);
#if NORMALIZE_VARIANCE == 1
    in_vals_squared_mean = work_group_reduce_add(in_vals_squared_mean) / (SG_SIZE * DATA_SET_SIZE);
#endif

#if NORMALIZE_VARIANCE == 1
    inv_sqrt_var = in_vals_squared_mean - in_vals_mean * in_vals_mean;
#if defined EPS_OUTSIDE_SQRT
    inv_sqrt_var = native_powr(native_sqrt(inv_sqrt_var) + (ACCUMULATOR_TYPE)EPSILON, (ACCUMULATOR_TYPE)-1);
#elif defined EPS_INSIDE_SQRT
    inv_sqrt_var = native_powr(inv_sqrt_var + (ACCUMULATOR_TYPE)EPSILON, (ACCUMULATOR_TYPE)-0.5);
#endif
#endif

    __attribute__((opencl_unroll_hint))
    for (int y = 0; y < TILE_SIZE_Y; y++) {
        __attribute__((opencl_unroll_hint))
        for (int x = 0; x < TILE_SIZE_X; x++) {
#if NORMALIZE_VARIANCE == 1
            in_vals[y][x] = (in_vals[y][x] - in_vals_mean) * inv_sqrt_var;
#else
            in_vals[y][x] -= in_vals_mean;
#endif
            ACTIVATION_TYPE out_val = TO_ACTIVATION_TYPE(in_vals[y][x]);
#if HAS_FUSED_OPS
            FUSED_OPS;
            output[idx] = FUSED_OPS_RESULT;
#else
            output[idx] = TO_OUTPUT_TYPE(ACTIVATION(out_val, ACTIVATION_PARAMS));
#endif
            idx += INPUT0_X_PITCH;
        }
        idx += INPUT0_Y_PITCH - TILE_SIZE_X * INPUT0_X_PITCH;
    }
    idx -= TILE_SIZE_Y * INPUT0_Y_PITCH;
}
