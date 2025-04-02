// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

#if F_BLOCK_SIZE == 1
#define ACCUMULATOR_VECTOR_TYPE ACCUMULATOR_TYPE
#else
#define ACCUMULATOR_VECTOR_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, F_BLOCK_SIZE)
#endif

#if F_BLOCK_SIZE == 1
#define ACTIVATION_FUNC_VECTOR_TYPE ACTIVATION_FUNC_TYPE
#else
#define ACTIVATION_FUNC_VECTOR_TYPE MAKE_VECTOR_TYPE(ACTIVATION_FUNC_TYPE, F_BLOCK_SIZE)
#endif

#define TO_ACTIVATION_FUNC_VECTOR_TYPE(data) CAT(convert_, ACTIVATION_FUNC_VECTOR_TYPE)(data)

#if F_BLOCK_SIZE == 1
#define INPUT0_VECTOR_TYPE INPUT0_TYPE
#else
#define INPUT0_VECTOR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, F_BLOCK_SIZE)
#endif

#if F_BLOCK_SIZE == 1
#if INPUT0_TYPE_SIZE == 4
#define BLOCK_LOAD(start_elem_idx, buffer) buffer[start_elem_idx]
#define BLOCK_STORE(data, start_elem_idx, buffer) buffer[start_elem_idx] = data
#else
#define BLOCK_LOAD(start_elem_idx, buffer) vload_half(0, buffer + start_elem_idx)
#define BLOCK_STORE(data, start_elem_idx, buffer) vstore_half(data, 0, buffer + start_elem_idx)
#endif
#define BLOCK_LOAD_AND_CONVERT_TO_ACCUMULATOR_VECTOR_TYPE(start_elem_idx, buffer) BLOCK_LOAD(start_elem_idx, buffer)
#else
#define BLOCK_LOAD(start_elem_idx, buffer) CAT(vload, F_BLOCK_SIZE)(0, buffer + start_elem_idx)
#define BLOCK_STORE(data, start_elem_idx, buffer) CAT(vstore, F_BLOCK_SIZE)(data, 0, buffer + start_elem_idx)
#if INPUT0_TYPE_SIZE == 4
#define BLOCK_LOAD_AND_CONVERT_TO_ACCUMULATOR_VECTOR_TYPE(start_elem_idx, buffer) BLOCK_LOAD(start_elem_idx, buffer)
#else
#define BLOCK_LOAD_AND_CONVERT_TO_ACCUMULATOR_VECTOR_TYPE(start_elem_idx, buffer) CAT(vload_half, F_BLOCK_SIZE)(0, buffer + start_elem_idx)
#endif
#endif

#define INPUT0_VECTOR_TYPE_UNION \
union { \
    INPUT0_TYPE f[F_BLOCK_SIZE]; \
    INPUT0_VECTOR_TYPE v; \
}

#define ACCUMULATOR_VECTOR_TYPE_UNION \
union { \
    ACCUMULATOR_TYPE f[F_BLOCK_SIZE]; \
    ACCUMULATOR_VECTOR_TYPE v; \
}

#define ACTIVATION_FUNC_VECTOR_TYPE_UNION \
union { \
    ACTIVATION_FUNC_TYPE f[F_BLOCK_SIZE]; \
    ACTIVATION_FUNC_VECTOR_TYPE v; \
}

#if !IS_DYNAMIC
#ifdef WITHIN_CHANNELS
__attribute__((reqd_work_group_size(1, 1, SPATIAL_WG_SIZE)))
#elif defined(ACROSS_CHANNELS)
__attribute__((reqd_work_group_size(1, INPUT0_FEATURE_NUM / F_BLOCK_SIZE, SPATIAL_WG_SIZE)))
#endif
#endif
KERNEL(mvn_within_channel_byxf_opt)(
	OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint b_start = get_global_id(0);
    const uint f_start = (uint)get_global_id(1) * NUM_F_BLOCKS_PER_WI * F_BLOCK_SIZE;
	const uint sp_start = (uint)get_global_id(2);
	
	const uint x_start = (sp_start % INPUT0_SIZE_X) * NUM_X_PER_WI;
#if INPUT0_DIMS == 4
	const uint y_start = (sp_start / INPUT0_SIZE_X) * NUM_Y_PER_WI;
	const uint z_start = 0;
#else
	const uint zy_start = sp_start / INPUT0_SIZE_X;
	const uint y_start = (zy % INPUT0_SIZE_Y) * NUM_Y_PER_WI;
	const uint z_start = (zy / INPUT0_SIZE_Y) * NUM_Z_PER_WI;
#endif

#if INPUT0_DIMS == 5
	uint idx = GET_DATA_INDEX(INPUT0, b_start, f_start, z_start, y_start, x_start);
#else
	uint idx = GET_DATA_INDEX(INPUT0, b_start, f_start, y_start, x_start);
#endif

#ifdef WITHIN_CHANNELS
#if SPATIAL_WG_SIZE > 1
__local ACCUMULATOR_VECTOR_TYPE_UNION slm_in_vals_mean[SPATIAL_WG_SIZE / 2][NUM_F_BLOCKS_PER_WI];
__local ACCUMULATOR_VECTOR_TYPE_UNION slm_in_vals_squared_mean[SPATIAL_WG_SIZE/ 2][NUM_F_BLOCKS_PER_WI];
#endif
ACCUMULATOR_VECTOR_TYPE_UNION in_vals_mean[NUM_F_BLOCKS_PER_WI];
#if NORMALIZE_VARIANCE == 1
ACCUMULATOR_VECTOR_TYPE_UNION in_vals_squared_mean[NUM_F_BLOCKS_PER_WI];
ACCUMULATOR_VECTOR_TYPE_UNION inv_sqrt_var[NUM_F_BLOCKS_PER_WI];
#endif
for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
	in_vals_mean[fb].v = (ACCUMULATOR_TYPE)0;
#if NORMALIZE_VARIANCE == 1
	in_vals_squared_mean[fb].v = (ACCUMULATOR_TYPE)0;
	inv_sqrt_var[fb].v = (ACCUMULATOR_TYPE)0;
#endif
}
#elif defined(ACROSS_CHANNELS)
ACCUMULATOR_TYPE in_vals_mean = (ACCUMULATOR_TYPE)0;
#if NORMALIZE_VARIANCE == 1
ACCUMULATOR_TYPE in_vals_squared_mean = (ACCUMULATOR_TYPE)0;
ACCUMULATOR_TYPE inv_sqrt_var = (ACCUMULATOR_TYPE)0;
#endif
#endif

#if INPUT0_DIMS == 5
	for (int z = 0; z < NUM_Z_PER_WI; z++) {
#endif
		for (int y = 0; y < NUM_Y_PER_WI; y++) {
			for (int x = 0; x < NUM_X_PER_WI; x++) {
				for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
					ACCUMULATOR_VECTOR_TYPE_UNION in_vals;
					in_vals.v = BLOCK_LOAD_AND_CONVERT_TO_ACCUMULATOR_VECTOR_TYPE(idx, input);
#ifdef WITHIN_CHANNELS
					in_vals_mean[fb].v += in_vals.v;
#if NORMALIZE_VARIANCE == 1
					// in_vals_squared_mean[fb].v += in_vals.v * in_vals.v;
					in_vals_squared_mean[fb].v = fma(in_vals.v, in_vals.v, in_vals_squared_mean[fb].v); 
#endif
#elif defined(ACROSS_CHANNELS)
					for (int i=0; i<F_BLOCK_SIZE; i++) {
						in_vals_mean += in_vals.f[i];
#if NORMALIZE_VARIANCE == 1
						// in_vals_squared_mean += in_vals.f[i] * in_vals.f[i];
						in_vals_squared_mean = fma(in_vals.f[i], in_vals.f[i], in_vals_squared_mean); 
#endif
					}
#endif	
					idx += F_BLOCK_SIZE;
				}
				idx += INPUT0_X_PITCH - NUM_F_BLOCKS_PER_WI * F_BLOCK_SIZE;
			}
			idx += INPUT0_Y_PITCH - NUM_X_PER_WI * INPUT0_X_PITCH;
		}
#if INPUT0_DIMS == 5
		idx += INPUT0_Z_PITCH - NUM_Y_PER_WI * INPUT0_Y_PITCH;
	}
	idx -= NUM_Z_PER_WI * INPUT0_Z_PITCH;
#else
		idx -= NUM_Y_PER_WI * INPUT0_Y_PITCH;
#endif

#ifdef WITHIN_CHANNELS
#if SPATIAL_WG_SIZE > 1
	const uint tid = sp_start;
	for (int stride = 2; stride < 2*SPATIAL_WG_SIZE; stride *= 2) {
        if (tid % stride == stride/2) {
			for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
				slm_in_vals_mean[tid/stride][fb].v = in_vals_mean[fb].v; 
#if NORMALIZE_VARIANCE == 1
				slm_in_vals_squared_mean[tid/stride][fb].v = in_vals_squared_mean[fb].v; 
#endif
			}
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid % stride == 0) && (tid + stride/2 < SPATIAL_WG_SIZE)) {
            for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
				in_vals_mean[fb].v += slm_in_vals_mean[tid/stride][fb].v; 
#if NORMALIZE_VARIANCE == 1
				in_vals_squared_mean[fb].v += slm_in_vals_squared_mean[tid/stride][fb].v; 
#endif
			}
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
		for (int i = 0; i<F_BLOCK_SIZE; i++) {
			in_vals_mean[fb].f[i] = work_group_broadcast(in_vals_mean[fb].f[i], 0) / DATA_SET_SIZE;
#if NORMALIZE_VARIANCE == 1
			in_vals_squared_mean[fb].f[i] = work_group_broadcast(in_vals_squared_mean[fb].f[i], 0) / DATA_SET_SIZE;
#endif
		}
	}
#else
	for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
		for (int i = 0; i<F_BLOCK_SIZE; i++) {
			in_vals_mean[fb].f[i] /= DATA_SET_SIZE;
#if NORMALIZE_VARIANCE == 1
			in_vals_squared_mean[fb].f[i] /= DATA_SET_SIZE;
#endif
		}
	}
#endif
#elif defined(ACROSS_CHANNELS)
	in_vals_mean = work_group_reduce_add(in_vals_mean) / DATA_SET_SIZE;
#if NORMALIZE_VARIANCE == 1
	in_vals_squared_mean = work_group_reduce_add(in_vals_squared_mean) / DATA_SET_SIZE;
#endif
#endif

#if NORMALIZE_VARIANCE == 1
#ifdef WITHIN_CHANNELS
	for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
		inv_sqrt_var[fb].v = in_vals_squared_mean[fb].v - in_vals_mean[fb].v * in_vals_mean[fb].v;	
#if defined EPS_OUTSIDE_SQRT
        inv_sqrt_var[fb].v = native_powr(native_sqrt(inv_sqrt_var[fb].v) + (ACCUMULATOR_TYPE)EPSILON, (ACCUMULATOR_TYPE)-1);
#elif defined EPS_INSIDE_SQRT
        inv_sqrt_var[fb].v = native_powr(inv_sqrt_var[fb].v + (ACCUMULATOR_TYPE)EPSILON, (ACCUMULATOR_TYPE)-0.5);
#endif
	}
#elif defined(ACROSS_CHANNELS)
	inv_sqrt_var = in_vals_squared_mean - in_vals_mean * in_vals_mean;
#if defined EPS_OUTSIDE_SQRT
    inv_sqrt_var = native_powr(native_sqrt(inv_sqrt_var) + (ACCUMULATOR_TYPE)EPSILON, (ACCUMULATOR_TYPE)-1);
#elif defined EPS_INSIDE_SQRT
    inv_sqrt_var = native_powr(inv_sqrt_var + (ACCUMULATOR_TYPE)EPSILON, (ACCUMULATOR_TYPE)-0.5);
#endif
#endif
#endif

#if INPUT0_DIMS == 5
	for (int z = 0; z < NUM_Z_PER_WI; z++) {
#endif
		for (int y = 0; y < NUM_Y_PER_WI; y++) {
			for (int x = 0; x < NUM_X_PER_WI; x++) {
				for (int fb = 0; fb < NUM_F_BLOCKS_PER_WI; fb++) {
					ACCUMULATOR_VECTOR_TYPE_UNION out_vals_acc;
					out_vals_acc.v = BLOCK_LOAD_AND_CONVERT_TO_ACCUMULATOR_VECTOR_TYPE(idx, input);
#ifdef WITHIN_CHANNELS
#if NORMALIZE_VARIANCE == 1
					out_vals_acc.v = (out_vals_acc.v - in_vals_mean[fb].v) * inv_sqrt_var[fb].v;
#else
					out_vals_acc.v -= in_vals_mean[fb].v;
#endif
#elif defined(ACROSS_CHANNELS)
#if NORMALIZE_VARIANCE == 1
					out_vals_acc.v = (out_vals_acc.v - in_vals_mean) * inv_sqrt_var;
#else
					out_vals_acc.v -= in_vals_mean;
#endif
#endif
					ACTIVATION_FUNC_VECTOR_TYPE_UNION out_vals;
					out_vals.v = TO_ACTIVATION_FUNC_VECTOR_TYPE(out_vals_acc.v);
					uint store_idx = idx;	
#if HAS_FUSED_OPS
					for (int i = 0; i < F_BLOCK_SIZE; i++) {
						FUSED_OPS;
						out_vals.f[i] = FUSED_OPS_RESULT;
						idx++;
					}
#else
					idx += F_BLOCK_SIZE;
#endif
					BLOCK_STORE(out_vals.v, store_idx, output);
				}
				idx += INPUT0_X_PITCH - NUM_F_BLOCKS_PER_WI * F_BLOCK_SIZE;
			}
			idx += INPUT0_Y_PITCH - NUM_X_PER_WI * INPUT0_X_PITCH;
		}
#if INPUT0_DIMS == 5
		idx += INPUT0_Z_PITCH - NUM_Y_PER_WI * INPUT0_Y_PITCH;
	}
	idx -= NUM_Z_PER_WI * INPUT0_Z_PITCH;
#else
		idx -= NUM_Y_PER_WI * INPUT0_Y_PITCH;
#endif
}

#undef ACCUMULATOR_VECTOR_TYPE
#undef ACTIVATION_FUNC_VECTOR_TYPE
#undef TO_ACTIVATION_FUNC_VECTOR_TYPE
#undef INPUT0_VECTOR_TYPE
#undef BLOCK_LOAD
#undef BLOCK_STORE
#undef BLOCK_LOAD_AND_CONVERT_TO_ACCUMULATOR_VECTOR_TYPE
#undef INPUT0_VECTOR_TYPE_UNION
#undef ACCUMULATOR_VECTOR_TYPE_UNION
#undef ACTIVATION_FUNC_VECTOR_TYPE_UNION
