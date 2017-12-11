/*
 *  BeagleCPU4StateAVXImpl.cpp
 *  BEAGLE
 *
 * Copyright 2013 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 */

#ifndef BEAGLE_CPU_4STATE_AVX_IMPL_HPP
#define BEAGLE_CPU_4STATE_AVX_IMPL_HPP


#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPU4StateAVXImpl.h"
#include "libhmsbeagle/CPU/AVXDefinitions.h"

template<int K>
inline V_Real pick_single(V_Real x) {
	V_Real t = _mm256_permute2f128_pd(x,x, K&2?49:32);
	return _mm256_permute_pd(t,K&1?15:0);
}


/* Loads partials into AVX vectors */
#if 0
#define AVX_PREFETCH_PARTIALS(dest, src, v) \
		dest##0 = _mm256_broadcast_sd(&src[v + 0]); \
		dest##1 = _mm256_broadcast_sd(&src[v + 1]); \
		dest##2 = _mm256_broadcast_sd(&src[v + 2]); \
		dest##3 = _mm256_broadcast_sd(&src[v + 3]);
#else // Load four partials in one 256-bit memory transactions
#define AVX_PREFETCH_PARTIALS(dest, src, v) \
		V_Real tmp_##dest##0123, permute_##dest##_01, permute_##dest##_23; \
		tmp_##dest##0123 = _mm256_load_pd(&src[v + 0]); \
		permute_##dest##_01 = _mm256_permute2f128_pd(tmp_##dest##0123, tmp_##dest##0123, 32); \
		permute_##dest##_23 = _mm256_permute2f128_pd(tmp_##dest##0123, tmp_##dest##0123, 49); \
		dest##0 = _mm256_permute_pd(permute_##dest##_01, 0); \
		dest##1 = _mm256_permute_pd(permute_##dest##_01, 15); \
		dest##2 = _mm256_permute_pd(permute_##dest##_23, 0); \
		dest##3 = _mm256_permute_pd(permute_##dest##_23, 15);
//		dest##0 = pick_single<0>(tmp_##dest##0123); \
//		dest##1 = pick_single<1>(tmp_##dest##0123); \
//		dest##2 = pick_single<2>(tmp_##dest##0123); \
//		dest##3 = pick_single<3>(tmp_##dest##0123);
#endif

// TODO Rearrange for optimal performance
/* Loads (transposed) finite-time transition matrices into AVX vectors */
#define AVX_PREFETCH_MATRICES(src_m1, src_m2, dest_vu_m1, dest_vu_m2) \
	const double *m1 = (src_m1); \
	const double *m2 = (src_m2); \
	for (int i = 0; i < OFFSET; i++, m1++, m2++) { \
		dest_vu_m1[i].x[0] = m1[0*OFFSET]; \
		dest_vu_m1[i].x[1] = m1[1*OFFSET]; \
		dest_vu_m1[i].x[2] = m1[2*OFFSET]; \
		dest_vu_m1[i].x[3] = m1[3*OFFSET]; \
		dest_vu_m2[i].x[0] = m2[0*OFFSET]; \
		dest_vu_m2[i].x[1] = m2[1*OFFSET]; \
		dest_vu_m2[i].x[2] = m2[2*OFFSET]; \
		dest_vu_m2[i].x[3] = m2[3*OFFSET]; \
	}

#define AVX_PREFETCH_MATRIX(src_m1, dest_vu_m1) \
	const double *m1 = (src_m1); \
	for (int i = 0; i < OFFSET; i++, m1++) { \
		dest_vu_m1[i][0].x[0] = m1[0*OFFSET]; \
		dest_vu_m1[i][0].x[1] = m1[1*OFFSET]; \
		dest_vu_m1[i][1].x[0] = m1[2*OFFSET]; \
		dest_vu_m1[i][1].x[1] = m1[3*OFFSET]; \
	}

namespace beagle {
namespace cpu {


BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPU4StateAVXName(){ return "CPU-4State-AVX-Unknown"; };

template<>
inline const char* getBeagleCPU4StateAVXName<double>(){ return "CPU-4State-AVX-Double"; };

template<>
inline const char* getBeagleCPU4StateAVXName<float>(){ return "CPU-4State-AVX-Single"; };
    
/*
 * Calculates partial likelihoods at a node when both children have states.
 */

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcStatesStates(float* destP,
                                     const int* states_q,
                                     const float* matrices_q,
                                     const int* states_r,
                                     const float* matrices_r) {

									 BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcStatesStates(destP,
                                     states_q,
                                     matrices_q,
                                     states_r,
                                     matrices_r);

									 }



BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesStates(double* destP,
                                     const int* states_q,
                                     const double* matrices_q,
                                     const int* states_r,
                                     const double* matrices_r) {

//	BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesStates(destP, states_q, matrices_q, states_r, matrices_r);
//	return;

	__m256d cache[OFFSET * OFFSET];
	__m256d * dest = (__m256d *) destP;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int w = l*4*OFFSET;
        int u = 0;
        int x = 0;
        for (int state2 = 0; state2 < OFFSET; state2++) {
        		register const __m256d ymm2 = _mm256_setr_pd(
                    matrices_r[w            + state2],
                    matrices_r[w + OFFSET*1 + state2],
                    matrices_r[w + OFFSET*2 + state2],
                    matrices_r[w + OFFSET*3 + state2]
            		);
        		for (int state1 = 0; state1 < OFFSET; state1++) {
        			register const __m256d ymm1 = _mm256_setr_pd(
                            matrices_q[w            + state1],
                            matrices_q[w + OFFSET*1 + state1],
                            matrices_q[w + OFFSET*2 + state1],
                            matrices_q[w + OFFSET*3 + state1]
                    		);
                    cache[x++] = _mm256_mul_pd(ymm2, ymm1);
        		}
        }
		for (int k = 0; k < kPatternCount; k++) {
			const int state1 = states_q[k];
			const int state2 = states_r[k];
			*dest++ = cache[state1 * OFFSET + state2];
        }
		if (kExtraPatterns)
			dest += kExtraPatterns * 2;
    }

//	__m256d * dest = (__m256d *) destP;
//#pragma omp parallel for num_threads(kCategoryCount)
//    for (int l = 0; l < kCategoryCount; l++) {
////        int v = l*4*kPaddedPatternCount;
//        int w = l*4*OFFSET;
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//            const int state1 = states_q[k];
//            const int state2 = states_r[k];
//
//            const __m256d vec1 = _mm256_setr_pd(
//                    matrices_q[w            + state1],
//                    matrices_q[w + OFFSET*1 + state1],
//                    matrices_q[w + OFFSET*2 + state1],
//                    matrices_q[w + OFFSET*3 + state1]
//            		);
//            const __m256d vec2 = _mm256_setr_pd(
//                    matrices_r[w            + state2],
//                    matrices_r[w + OFFSET*1 + state2],
//                    matrices_r[w + OFFSET*2 + state2],
//                    matrices_r[w + OFFSET*3 + state2]
//            		);
//
//            *dest++ = _mm256_mul_pd(vec1, vec2);
//        }
//        if (kExtraPatterns)
//        		dest += kExtraPatterns * 2;
//    }
//    return;
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesStatesFixedScaling(double* destP,
                                     const int* states_q,
                                     const double* matrices_q,
                                     const int* states_r,
                                     const double* matrices_r,
									const double* __restrict scaleFactors) {

//	BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesStatesFixedScaling(destP, states_q, matrices_q, states_r, matrices_r, scaleFactors);
//	return;

	__m256d cache[OFFSET * OFFSET];
	__m256d * dest = (__m256d *) destP;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int w = l*4*OFFSET;
        int u = 0;
        int x = 0;
        for (int state2 = 0; state2 < OFFSET; state2++) {
            register const __m256d ymm2 = _mm256_setr_pd(
                    matrices_r[w            + state2],
                    matrices_r[w + OFFSET*1 + state2],
                    matrices_r[w + OFFSET*2 + state2],
                    matrices_r[w + OFFSET*3 + state2]
            		);
        		for (int state1 = 0; state1 < OFFSET; state1++) {
        			register const __m256d ymm1 = _mm256_setr_pd(
                            matrices_q[w            + state1],
                            matrices_q[w + OFFSET*1 + state1],
                            matrices_q[w + OFFSET*2 + state1],
                            matrices_q[w + OFFSET*3 + state1]
                    		);
                    cache[x++] = _mm256_mul_pd(ymm2, ymm1);
        		}
        }
		for (int k = 0; k < kPatternCount; k++) {
			const int state1 = states_q[k];
			const int state2 = states_r[k];
			*dest++ = _mm256_div_pd(cache[state1 * OFFSET + state2], _mm256_set1_pd(scaleFactors[k]));
        }
		if (kExtraPatterns)
			dest += kExtraPatterns * 2;
    }

//	__m256d * dest = (__m256d *) destP;
//#pragma omp parallel for num_threads(kCategoryCount)
//    for (int l = 0; l < kCategoryCount; l++) {
//        int w = l*4*OFFSET;
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//            const int state1 = states_q[k];
//            const int state2 = states_r[k];
//
//            const __m256d vec1 = _mm256_setr_pd(
//                    matrices_q[w            + state1],
//                    matrices_q[w + OFFSET*1 + state1],
//                    matrices_q[w + OFFSET*2 + state1],
//                    matrices_q[w + OFFSET*3 + state1]
//            		);
//            const __m256d vec2 = _mm256_setr_pd(
//                    matrices_r[w            + state2],
//                    matrices_r[w + OFFSET*1 + state2],
//                    matrices_r[w + OFFSET*2 + state2],
//                    matrices_r[w + OFFSET*3 + state2]
//            		);
//
//            __m256d unscaled = _mm256_mul_pd(vec1, vec2);
//            *dest++ = _mm256_div_pd(unscaled, _mm256_set1_pd(scaleFactors[k]));
//        }
//        if (kExtraPatterns)
//        		dest += kExtraPatterns * 2;
//    }
//    return;
}





/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   AVX version
 */
BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcStatesPartials(float* destP,
                                       const int* states_q,
                                       const float* matrices_q,
                                       const float* partials_r,
                                       const float* matrices_r) {
	BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcStatesPartials(
									   destP,
									   states_q,
									   matrices_q,
									   partials_r,
									   matrices_r);
}



BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesPartials(double* destP,
                                       const int* states_q,
                                       const double* matrices_q,
                                       const double* partials_r,
                                       const double* matrices_r) {
//	BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesPartials(destP, states_q, matrices_q, partials_r, matrices_r);
//	return;
	__m256d * dest = (__m256d *) destP;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
//        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;

        // PREFETCH_MATRIX(2,matrices_r,w);
//        double m200, m201, m202, m203,
//               m210, m211, m212, m213,
//               m220, m221, m222, m223,
//               m230, m231, m232, m233;
//        m200 = matrices_r[w + OFFSET*0 + 0];
//        m201 = matrices_r[w + OFFSET*0 + 1];
//        m202 = matrices_r[w + OFFSET*0 + 2];
//        m203 = matrices_r[w + OFFSET*0 + 3];
//        m210 = matrices_r[w + OFFSET*1 + 0];
//        m211 = matrices_r[w + OFFSET*1 + 1];
//        m212 = matrices_r[w + OFFSET*1 + 2];
//        m213 = matrices_r[w + OFFSET*1 + 3];
//        m220 = matrices_r[w + OFFSET*2 + 0];
//        m221 = matrices_r[w + OFFSET*2 + 1];
//        m222 = matrices_r[w + OFFSET*2 + 2];
//        m223 = matrices_r[w + OFFSET*2 + 3];
//        m230 = matrices_r[w + OFFSET*3 + 0];
//        m231 = matrices_r[w + OFFSET*3 + 1];
//        m232 = matrices_r[w + OFFSET*3 + 2];
//        m233 = matrices_r[w + OFFSET*3 + 3];

//        __m256d  * matrix20 = (__m256d *) (matrices_r  + w);
//        __m256d  * matrix21 = (__m256d *) (matrices_r  + w + OFFSET*1);
//        __m256d  * matrix22 = (__m256d *) (matrices_r  + w + OFFSET*2);
//        __m256d  * matrix23 = (__m256d *) (matrices_r  + w + OFFSET*3);


        const __m256d ymm0 = _mm256_setr_pd(
                matrices_r[w            ],
                matrices_r[w + OFFSET*1 ],
                matrices_r[w + OFFSET*2 ],
                matrices_r[w + OFFSET*3 ]
        		);
        const __m256d ymm1 = _mm256_setr_pd(
                matrices_r[w            + 1],
                matrices_r[w + OFFSET*1 + 1],
                matrices_r[w + OFFSET*2 + 1],
                matrices_r[w + OFFSET*3 + 1]
        		);
        const __m256d ymm2 = _mm256_setr_pd(
                matrices_r[w            + 2],
                matrices_r[w + OFFSET*1 + 2],
                matrices_r[w + OFFSET*2 + 2],
                matrices_r[w + OFFSET*3 + 2]
        		);
        const __m256d ymm3 = _mm256_setr_pd(
                matrices_r[w            + 3],
                matrices_r[w + OFFSET*1 + 3],
                matrices_r[w + OFFSET*2 + 3],
                matrices_r[w + OFFSET*3 + 3]
        		);

        __m256d matrixQ[5];
        matrixQ[0] = _mm256_setr_pd(
                matrices_q[w            ],
                matrices_q[w + OFFSET*1 ],
                matrices_q[w + OFFSET*2 ],
                matrices_q[w + OFFSET*3 ]
        		);
        matrixQ[1] = _mm256_setr_pd(
                matrices_q[w            + 1],
                matrices_q[w + OFFSET*1 + 1],
                matrices_q[w + OFFSET*2 + 1],
                matrices_q[w + OFFSET*3 + 1]
        		);
        matrixQ[2] = _mm256_setr_pd(
                matrices_q[w            + 2],
                matrices_q[w + OFFSET*1 + 2],
                matrices_q[w + OFFSET*2 + 2],
                matrices_q[w + OFFSET*3 + 2]
        		);
        matrixQ[3] = _mm256_setr_pd(
                matrices_q[w            + 3],
                matrices_q[w + OFFSET*1 + 3],
                matrices_q[w + OFFSET*2 + 3],
                matrices_q[w + OFFSET*3 + 3]
        		);
        matrixQ[4] = _mm256_set1_pd(1.0);



//        __m256d * p = (__m256d *) (partials_r);
        const double * p = partials_r;
        for (int k = 0; k < kPatternCount; k++) {

            //PREFETCH_PARTIALS(2,partials_r,u);
//            double p20, p21, p22, p23;
//                p20 = partials_r[u + 0];
//                p21 = partials_r[u + 1];
//                p22 = partials_r[u + 2];
//                p23 = partials_r[u + 3];

            // DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23;
//            double sum20, sum21, sum22, sum23; \
//            sum20  = m200 * p20 +
//                           m201 * p21 +
//                           m202 * p22 +
//                           m203 * p23;
//
//            sum21  = m210 * p20 +
//                           m211 * p21 +
//                           m212 * p22 +
//                           m213 * p23;
//
//            sum22  = m220 * p20 +
//                           m221 * p21 +
//                           m222 * p22 +
//                           m223 * p23;
//
//            sum23  = m230 * p20 +
//                           m231 * p21 +
//                           m232 * p22 +
//                           m233 * p23;


            __m256d sum = _mm256_mul_pd(_mm256_set1_pd(*p++), ymm0);
            sum = _mm256_fmadd_pd(_mm256_set1_pd(*p++), ymm1, sum);
            sum = _mm256_fmadd_pd(_mm256_set1_pd(*p++), ymm2, sum);
            sum = _mm256_fmadd_pd(_mm256_set1_pd(*p++), ymm3, sum);




//            __m256d sum0 = _mm256_mul_pd(*p, *matrix20);
//            __m256d sum1 = _mm256_mul_pd(*p, *matrix21);
//            __m256d sum2 = _mm256_mul_pd(*p, *matrix22);
//            __m256d sum3 = _mm256_mul_pd(*p, *matrix23);
//
//            __m256d sum01 = _mm256_hadd_pd(sum0, sum1);
//            // extract upper 128 bits of result
//            __m128d sum_high01 = _mm256_extractf128_pd(sum01, 1);
//            // add upper 128 bits of sum to its lower 128 bits
//            __m128d result01 = _mm_add_pd(sum_high01, _mm256_castpd256_pd128(sum01));
//
//            __m256d sum23 = _mm256_hadd_pd(sum2, sum3);
//            // extract upper 128 bits of result
//            __m128d sum_high23 = _mm256_extractf128_pd(sum23, 1);
//            // add upper 128 bits of sum to its lower 128 bits
//            __m128d result23 = _mm_add_pd(sum_high23, _mm256_castpd256_pd128(sum23));

//            destP[u    ] = matrices_q[w            + state1] * sum20;
//            destP[u + 1] = matrices_q[w + OFFSET*1 + state1] * sum21;
//            destP[u + 2] = matrices_q[w + OFFSET*2 + state1] * sum22;
//            destP[u + 3] = matrices_q[w + OFFSET*3 + state1] * sum23;

            const int state1 = states_q[k];
//            const __m256d matrix1 = _mm256_setr_pd(
//                    matrices_q[w            + state1],
//                    matrices_q[w + OFFSET*1 + state1],
//                    matrices_q[w + OFFSET*2 + state1],
//                    matrices_q[w + OFFSET*3 + state1]
//            		);
//             __m256d sum = _mm256_setr_pd(
//                    result01[0],
//                    result01[1],
//                    result23[0],
//                    result23[1]
//            		);

            *dest++ = _mm256_mul_pd(matrixQ[state1], sum);

//            u += 4;
//            p++;
        }
        if (kExtraPatterns)
        		dest += kExtraPatterns * 2;
    }

    return;
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcStatesPartialsFixedScaling(float* destP,
                                const int* states1,
                                const float* __restrict matrices1,
                                const float* __restrict partials2,
                                const float* __restrict matrices2,
                                const float* __restrict scaleFactors) {
	BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcStatesPartialsFixedScaling(
									   destP,
									   states1,
									   matrices1,
									   partials2,
									   matrices2,
									   scaleFactors);
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesPartialsFixedScaling(double* destP,
                                const int* states_q,
                                const double* __restrict matrices_q,
                                const double* __restrict partials_r,
                                const double* __restrict matrices_r,
                                const double* __restrict scaleFactors) {
//	BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcStatesPartialsFixedScaling(destP, states_q, matrices_q, partials_r, matrices_r, scaleFactors);
//	return;
	__m256d * dest = (__m256d *) destP;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int w = l*4*OFFSET;

        const __m256d ymm0 = _mm256_setr_pd(
                matrices_r[w            ],
                matrices_r[w + OFFSET*1 ],
                matrices_r[w + OFFSET*2 ],
                matrices_r[w + OFFSET*3 ]
        		);
        const __m256d ymm1 = _mm256_setr_pd(
                matrices_r[w            + 1],
                matrices_r[w + OFFSET*1 + 1],
                matrices_r[w + OFFSET*2 + 1],
                matrices_r[w + OFFSET*3 + 1]
        		);
        const __m256d ymm2 = _mm256_setr_pd(
                matrices_r[w            + 2],
                matrices_r[w + OFFSET*1 + 2],
                matrices_r[w + OFFSET*2 + 2],
                matrices_r[w + OFFSET*3 + 2]
        		);
        const __m256d ymm3 = _mm256_setr_pd(
                matrices_r[w            + 3],
                matrices_r[w + OFFSET*1 + 3],
                matrices_r[w + OFFSET*2 + 3],
                matrices_r[w + OFFSET*3 + 3]
        		);

        const double * p = partials_r;
        for (int k = 0; k < kPatternCount; k++) {
            __m256d ymm4 = _mm256_mul_pd(_mm256_set1_pd(*p++), ymm0);
            ymm4 = _mm256_fmadd_pd(_mm256_set1_pd(*p++), ymm1, ymm4);
            ymm4 = _mm256_fmadd_pd(_mm256_set1_pd(*p++), ymm2, ymm4);
            ymm4 = _mm256_fmadd_pd(_mm256_set1_pd(*p++), ymm3, ymm4);

            const int state1 = states_q[k];
            const __m256d matrix1 = _mm256_setr_pd(
                    matrices_q[w            + state1],
                    matrices_q[w + OFFSET*1 + state1],
                    matrices_q[w + OFFSET*2 + state1],
                    matrices_q[w + OFFSET*3 + state1]
            		);

        		ymm4 = _mm256_mul_pd(matrix1, ymm4);
            *dest++ = _mm256_div_pd(ymm4, _mm256_set1_pd(scaleFactors[k]));
        }
        if (kExtraPatterns)
        		dest += kExtraPatterns * 2;
    }

    return;
//
//
//    int v = 0;
//    int w = 0;
//
// 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
//	V_Real *destPvec = (V_Real *)destP;
//	V_Real destr_01, destr_23;
//
//    for (int l = 0; l < kCategoryCount; l++) {
//
//    	//AVX_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//        	const V_Real scaleFactor = VEC_SPLAT(scaleFactors[k]);
//
//            const int state_q = states_q[k];
//            V_Real vp0, vp1, vp2, vp3;
//            AVX_PREFETCH_PARTIALS(vp,partials_r,v);
//
//			destr_01 = VEC_MULT(vp0, vu_mr[0][0].vx);
//			destr_01 = VEC_MADD(vp1, vu_mr[1][0].vx, destr_01);
//			destr_01 = VEC_MADD(vp2, vu_mr[2][0].vx, destr_01);
//			destr_01 = VEC_MADD(vp3, vu_mr[3][0].vx, destr_01);
//			destr_23 = VEC_MULT(vp0, vu_mr[0][1].vx);
//			destr_23 = VEC_MADD(vp1, vu_mr[1][1].vx, destr_23);
//			destr_23 = VEC_MADD(vp2, vu_mr[2][1].vx, destr_23);
//			destr_23 = VEC_MADD(vp3, vu_mr[3][1].vx, destr_23);
//
//            *destPvec++ = VEC_DIV(VEC_MULT(vu_mq[state_q][0].vx, destr_01), scaleFactor);
//            *destPvec++ = VEC_DIV(VEC_MULT(vu_mq[state_q][1].vx, destr_23), scaleFactor);
//
//            v += 4;
//        }
//        w += OFFSET*4;
//        if (kExtraPatterns) {
//        	destPvec += kExtraPatterns * 2;
//        	v += kExtraPatterns * 4;
//        }
//    }
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcPartialsPartials(float* destP,
                                                  const float*  partials_q,
                                                  const float*  matrices_q,
                                                  const float*  partials_r,
                                                  const float*  matrices_r) {

	BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcPartialsPartials(destP,
                                                  partials_q,
                                                  matrices_q,
                                                  partials_r,
                                                  matrices_r);
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcPartialsPartials(double* destP,
                                                  const double*  partials_q,
                                                  const double*  matrices_q,
                                                  const double*  partials_r,
                                                  const double*  matrices_r) {

	__m256d * dest = (__m256d *) destP;
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
//        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;

        //PREFETCH_MATRIX(1,matrices_q,w);
        const __m256d ymm10 = _mm256_setr_pd(
                matrices_q[w            ],
                matrices_q[w + OFFSET*1 ],
                matrices_q[w + OFFSET*2 ],
                matrices_q[w + OFFSET*3 ]
        		);
        const __m256d ymm11 = _mm256_setr_pd(
                matrices_q[w            + 1],
                matrices_q[w + OFFSET*1 + 1],
                matrices_q[w + OFFSET*2 + 1],
                matrices_q[w + OFFSET*3 + 1]
        		);
        const __m256d ymm12 = _mm256_setr_pd(
                matrices_q[w            + 2],
                matrices_q[w + OFFSET*1 + 2],
                matrices_q[w + OFFSET*2 + 2],
                matrices_q[w + OFFSET*3 + 2]
        		);
        const __m256d ymm13 = _mm256_setr_pd(
                matrices_q[w            + 3],
                matrices_q[w + OFFSET*1 + 3],
                matrices_q[w + OFFSET*2 + 3],
                matrices_q[w + OFFSET*3 + 3]
        		);
        //PREFETCH_MATRIX(2,matrices_r,w);
        const __m256d ymm20 = _mm256_setr_pd(
                matrices_r[w            ],
                matrices_r[w + OFFSET*1 ],
                matrices_r[w + OFFSET*2 ],
                matrices_r[w + OFFSET*3 ]
        		);
        const __m256d ymm21 = _mm256_setr_pd(
                matrices_r[w            + 1],
                matrices_r[w + OFFSET*1 + 1],
                matrices_r[w + OFFSET*2 + 1],
                matrices_r[w + OFFSET*3 + 1]
        		);
        const __m256d ymm22 = _mm256_setr_pd(
                matrices_r[w            + 2],
                matrices_r[w + OFFSET*1 + 2],
                matrices_r[w + OFFSET*2 + 2],
                matrices_r[w + OFFSET*3 + 2]
        		);
        const __m256d ymm23 = _mm256_setr_pd(
                matrices_r[w            + 3],
                matrices_r[w + OFFSET*1 + 3],
                matrices_r[w + OFFSET*2 + 3],
                matrices_r[w + OFFSET*3 + 3]
        		);
        const double * pq = partials_q;
        const double * pr = partials_r;
        for (int k = 0; k < kPatternCount; k++) {
            //PREFETCH_PARTIALS(1,partials_q,u);
            //PREFETCH_PARTIALS(2,partials_r,u);

            //DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            __m256d sumq = _mm256_mul_pd(_mm256_set1_pd(*pq++), ymm10);
            sumq = _mm256_fmadd_pd(_mm256_set1_pd(*pq++), ymm11, sumq);
            sumq = _mm256_fmadd_pd(_mm256_set1_pd(*pq++), ymm12, sumq);
            sumq = _mm256_fmadd_pd(_mm256_set1_pd(*pq++), ymm13, sumq);
            //DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            __m256d sumr = _mm256_mul_pd(_mm256_set1_pd(*pr++), ymm20);
            sumr = _mm256_fmadd_pd(_mm256_set1_pd(*pr++), ymm21, sumr);
            sumr = _mm256_fmadd_pd(_mm256_set1_pd(*pr++), ymm22, sumr);
            sumr = _mm256_fmadd_pd(_mm256_set1_pd(*pr++), ymm23, sumr);


            // Final results
            *dest++ = _mm256_mul_pd(sumq, sumr);
            //*dest++ = _mm256_div_pd(sum, _mm256_set1_pd(scaleFactors[k]));

            //u += 4;

        }
        if (kExtraPatterns)
        		dest += kExtraPatterns * 2;
    }
    return;


    int v = 0;
    int w = 0;

    V_Real	destq_0123, destr_0123;
 	VecUnion vu_mq[OFFSET], vu_mr[OFFSET];
	V_Real *destPvec = (V_Real *)destP;

//	for (int i = 0; i < 4; ++i) {
//		int t1 = i & 1;
//		int t2 = i & 2;
//		fprintf(stderr, "%d ->  %d %d\n", i, t1, t2);
//	}

    for (int l = 0; l < kCategoryCount; l++) {

		/* Load transition-probability matrices into vectors */
    	AVX_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

//    	for (int j = 0; j < 4; ++j) {
//    		fprintf(stderr, "mq[%d]:", j);
//    		VecUnion tmp;
//    		tmp.vx = vu_mq[j].vx;
//    		for (int i = 0; i < 4; ++i) {
//    			fprintf(stderr, " %5.3e", tmp.x[i]);
//    		}
//    		fprintf(stderr, "\n");
//    	}
//
//    	fprintf(stderr,"APM\n");

        for (int k = 0; k < kPatternCount; k++) {
            
#           if 1 && !defined(_WIN32)
            __builtin_prefetch (&partials_q[v+64]);
            __builtin_prefetch (&partials_r[v+64]);
#           endif

        	V_Real vpq_0, vpq_1, vpq_2, vpq_3;
        	AVX_PREFETCH_PARTIALS(vpq_,partials_q,v);

//        	fprintf(stderr, "t:");
//        	for (int i = 0; i < 4; ++i) {
//        		fprintf(stderr, " %5.3e", partials_q[v + i]);
//        	}
//        	fprintf(stderr, "\n");
//        	VecUnion tmp;
//        	fprintf(stderr, "g:");
//        	tmp.vx = vpq_0;
//        	for (int j = 0; j < 4; ++j) {
//        		fprintf(stderr, " %5.3f", tmp.x[j]);
//        	}
//        	tmp.vx = vpq_1;
//        	for (int j = 0; j < 4; ++j) {
//        		fprintf(stderr, " %5.3f", tmp.x[j]);
//        	}
//        	tmp.vx = vpq_2;
//        	for (int j = 0; j < 4; ++j) {
//        		fprintf(stderr, " %5.3f", tmp.x[j]);
//        	}
//        	tmp.vx = vpq_3;
//        	for (int j = 0; j < 4; ++j) {
//        		fprintf(stderr, " %5.3f", tmp.x[j]);
//        	}
//        	fprintf(stderr, "\n");

        	V_Real vpr_0, vpr_1, vpr_2, vpr_3;
        	AVX_PREFETCH_PARTIALS(vpr_,partials_r,v);

        	destq_0123 = VEC_MULT(vpq_0, vu_mq[0].vx);
        	destq_0123 = VEC_MADD(vpq_1, vu_mq[1].vx, destq_0123);
        	destq_0123 = VEC_MADD(vpq_2, vu_mq[2].vx, destq_0123);
        	destq_0123 = VEC_MADD(vpq_3, vu_mq[3].vx, destq_0123);

        	destr_0123 = VEC_MULT(vpr_0, vu_mr[0].vx);
        	destr_0123 = VEC_MADD(vpr_1, vu_mr[1].vx, destr_0123);
        	destr_0123 = VEC_MADD(vpr_2, vu_mr[2].vx, destr_0123);
        	destr_0123 = VEC_MADD(vpr_3, vu_mr[3].vx, destr_0123);

//        	*destPvec = VEC_MULT(destq_0123, destr_0123); // Single store
//        	destPvec += 1;

        	VEC_STORE(destP, VEC_MULT(destq_0123, destr_0123));
        	destP += 4;

//        	for (int i = 0; i < 4; ++i) {
//        		fprintf(stderr, " %5.3e", ((double*)destPvec)[i]);
//        	}
//        	fprintf(stderr, "\n");


            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
    }
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcPartialsPartialsFixedScaling(float* destP,
                                        const float*  child0Partials,
                                        const float*  child0TransMat,
                                        const float*  child1Partials,
                                        const float*  child1TransMat,
                                        const float*  scaleFactors) {

	BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcPartialsPartialsFixedScaling(
			destP,
			child0Partials,
			child0TransMat,
			child1Partials,
			child1TransMat,
			scaleFactors);
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcPartialsPartialsFixedScaling(double* destP,
		                                                        const double* partials_q,
		                                                        const double* matrices_q,
		                                                        const double* partials_r,
		                                                        const double* matrices_r,
		                                                        const double* scaleFactors) {

	__m256d * dest = (__m256d *) destP;
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
//        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;

        //PREFETCH_MATRIX(1,matrices_q,w);
        const __m256d ymm10 = _mm256_setr_pd(
                matrices_q[w            ],
                matrices_q[w + OFFSET*1 ],
                matrices_q[w + OFFSET*2 ],
                matrices_q[w + OFFSET*3 ]
        		);
        const __m256d ymm11 = _mm256_setr_pd(
                matrices_q[w            + 1],
                matrices_q[w + OFFSET*1 + 1],
                matrices_q[w + OFFSET*2 + 1],
                matrices_q[w + OFFSET*3 + 1]
        		);
        const __m256d ymm12 = _mm256_setr_pd(
                matrices_q[w            + 2],
                matrices_q[w + OFFSET*1 + 2],
                matrices_q[w + OFFSET*2 + 2],
                matrices_q[w + OFFSET*3 + 2]
        		);
        const __m256d ymm13 = _mm256_setr_pd(
                matrices_q[w            + 3],
                matrices_q[w + OFFSET*1 + 3],
                matrices_q[w + OFFSET*2 + 3],
                matrices_q[w + OFFSET*3 + 3]
        		);
        //PREFETCH_MATRIX(2,matrices_r,w);
        const __m256d ymm0 = _mm256_setr_pd(
                matrices_r[w            ],
                matrices_r[w + OFFSET*1 ],
                matrices_r[w + OFFSET*2 ],
                matrices_r[w + OFFSET*3 ]
        		);
        const __m256d ymm1 = _mm256_setr_pd(
                matrices_r[w            + 1],
                matrices_r[w + OFFSET*1 + 1],
                matrices_r[w + OFFSET*2 + 1],
                matrices_r[w + OFFSET*3 + 1]
        		);
        const __m256d ymm2 = _mm256_setr_pd(
                matrices_r[w            + 2],
                matrices_r[w + OFFSET*1 + 2],
                matrices_r[w + OFFSET*2 + 2],
                matrices_r[w + OFFSET*3 + 2]
        		);
        const __m256d ymm3 = _mm256_setr_pd(
                matrices_r[w            + 3],
                matrices_r[w + OFFSET*1 + 3],
                matrices_r[w + OFFSET*2 + 3],
                matrices_r[w + OFFSET*3 + 3]
        		);
        const double * pq = partials_q;
        const double * pr = partials_r;
        for (int k = 0; k < kPatternCount; k++) {
            //PREFETCH_PARTIALS(1,partials_q,u);
            //PREFETCH_PARTIALS(2,partials_r,u);

            //DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            __m256d ymm4 = _mm256_mul_pd(_mm256_set1_pd(*pq++), ymm10);
            ymm4 = _mm256_fmadd_pd(_mm256_set1_pd(*pq++), ymm11, ymm4);
            ymm4 = _mm256_fmadd_pd(_mm256_set1_pd(*pq++), ymm12, ymm4);
            ymm4 = _mm256_fmadd_pd(_mm256_set1_pd(*pq++), ymm13, ymm4);
            //DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            __m256d ymm5 = _mm256_mul_pd(_mm256_set1_pd(*pr++), ymm0);
            ymm5 = _mm256_fmadd_pd(_mm256_set1_pd(*pr++), ymm1, ymm5);
            ymm5 = _mm256_fmadd_pd(_mm256_set1_pd(*pr++), ymm2, ymm5);
            ymm5 = _mm256_fmadd_pd(_mm256_set1_pd(*pr++), ymm3, ymm5);


            // Final results
            ymm4 = _mm256_mul_pd(ymm4, ymm5);
            *dest++ = _mm256_div_pd(ymm4, _mm256_set1_pd(scaleFactors[k]));

            //u += 4;

        }
    }
    return;    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;

	for (int l = 0; l < kCategoryCount; l++) {

		/* Load transition-probability matrices into vectors */
    	//AVX_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = 0; k < kPatternCount; k++) {

#           if 1 && !defined(_WIN32)
            __builtin_prefetch (&partials_q[v+64]);
            __builtin_prefetch (&partials_r[v+64]);
            //            __builtin_prefetch (destPvec+32,1,0);
#           endif
            
            // Prefetch scale factor
//            const V_Real scaleFactor = VEC_LOAD_SCALAR(scaleFactors + k);
        	// Option below appears faster, why?
        	const V_Real scaleFactor = VEC_SPLAT(scaleFactors[k]);

        	V_Real vpq_0, vpq_1, vpq_2, vpq_3;
        	AVX_PREFETCH_PARTIALS(vpq_,partials_q,v);

        	V_Real vpr_0, vpr_1, vpr_2, vpr_3;
        	AVX_PREFETCH_PARTIALS(vpr_,partials_r,v);

        	// TODO Make below into macro since this repeats from other calcPPs
			destq_01 = VEC_MULT(vpq_0, vu_mq[0][0].vx);
			destq_01 = VEC_MADD(vpq_1, vu_mq[1][0].vx, destq_01);
			destq_01 = VEC_MADD(vpq_2, vu_mq[2][0].vx, destq_01);
			destq_01 = VEC_MADD(vpq_3, vu_mq[3][0].vx, destq_01);
			destq_23 = VEC_MULT(vpq_0, vu_mq[0][1].vx);
			destq_23 = VEC_MADD(vpq_1, vu_mq[1][1].vx, destq_23);
			destq_23 = VEC_MADD(vpq_2, vu_mq[2][1].vx, destq_23);
			destq_23 = VEC_MADD(vpq_3, vu_mq[3][1].vx, destq_23);

			destr_01 = VEC_MULT(vpr_0, vu_mr[0][0].vx);
			destr_01 = VEC_MADD(vpr_1, vu_mr[1][0].vx, destr_01);
			destr_01 = VEC_MADD(vpr_2, vu_mr[2][0].vx, destr_01);
			destr_01 = VEC_MADD(vpr_3, vu_mr[3][0].vx, destr_01);
			destr_23 = VEC_MULT(vpr_0, vu_mr[0][1].vx);
			destr_23 = VEC_MADD(vpr_1, vu_mr[1][1].vx, destr_23);
			destr_23 = VEC_MADD(vpr_2, vu_mr[2][1].vx, destr_23);
			destr_23 = VEC_MADD(vpr_3, vu_mr[3][1].vx, destr_23);

            destPvec[0] = VEC_DIV(VEC_MULT(destq_01, destr_01), scaleFactor);
            destPvec[1] = VEC_DIV(VEC_MULT(destq_23, destr_23), scaleFactor);

            destPvec += 2;
            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
    }
}

    
BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcPartialsPartialsAutoScaling(float* destP,
                                                         const float*  partials_q,
                                                         const float*  matrices_q,
                                                         const float*  partials_r,
                                                         const float*  matrices_r,
                                                                 int* activateScaling) {
    BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcPartialsPartialsAutoScaling(destP,
                                                     partials_q,
                                                     matrices_q,
                                                     partials_r,
                                                     matrices_r,
                                                     activateScaling);
}

BEAGLE_CPU_4_AVX_TEMPLATE
void BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcPartialsPartialsAutoScaling(double* destP,
                                                                    const double*  partials_q,
                                                                    const double*  matrices_q,
                                                                    const double*  partials_r,
                                                                    const double*  matrices_r,
                                                                    int* activateScaling) {
    // TODO: implement calcPartialsPartialsAutoScaling with AVX
    BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcPartialsPartialsAutoScaling(destP,
                                                                partials_q,
                                                                matrices_q,
                                                                partials_r,
                                                                matrices_r,
                                                                activateScaling);
}
    
BEAGLE_CPU_4_AVX_TEMPLATE
int BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcEdgeLogLikelihoods(const int parIndex,
                                                          const int childIndex,
                                                          const int probIndex,
                                                          const int categoryWeightsIndex,
                                                          const int stateFrequenciesIndex,
                                                          const int scalingFactorsIndex,
                                                          double* outSumLogLikelihood) {
    return BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT>::calcEdgeLogLikelihoods(
                                                              parIndex,
                                                              childIndex,
                                                              probIndex,
                                                              categoryWeightsIndex,
                                                              stateFrequenciesIndex,
                                                              scalingFactorsIndex,
                                                              outSumLogLikelihood);
}

BEAGLE_CPU_4_AVX_TEMPLATE
int BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::calcEdgeLogLikelihoods(const int parIndex,
                                                            const int childIndex,
                                                            const int probIndex,
                                                            const int categoryWeightsIndex,
                                                            const int stateFrequenciesIndex,
                                                            const int scalingFactorsIndex,
                                                            double* outSumLogLikelihood) {
    // TODO: implement derivatives for calculateEdgeLnL

    int returnCode = BEAGLE_SUCCESS;

    assert(parIndex >= kTipCount);

    const double* cl_r = gPartials[parIndex];
    double* cl_p = integrationTmp;
    const double* transMatrix = gTransitionMatrices[probIndex];
    const double* wt = gCategoryWeights[categoryWeightsIndex];
    const double* freqs = gStateFrequencies[stateFrequenciesIndex];

    memset(cl_p, 0, (kPatternCount * kStateCount)*sizeof(double));

    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

        const int* statesChild = gTipStates[childIndex];

        int w = 0;
        V_Real *vcl_r = (V_Real *)cl_r;
        for(int l = 0; l < kCategoryCount; l++) {

            VecUnion vu_m[OFFSET][2];
            AVX_PREFETCH_MATRIX(transMatrix + w, vu_m)

           V_Real *vcl_p = (V_Real *)cl_p;

           for(int k = 0; k < kPatternCount; k++) {

                const int stateChild = statesChild[k];
                V_Real vwt = VEC_SPLAT(wt[l]);

                V_Real wtdPartials = VEC_MULT(*vcl_r++, vwt);
                *vcl_p++ = VEC_MADD(vu_m[stateChild][0].vx, wtdPartials, *vcl_p);

                wtdPartials = VEC_MULT(*vcl_r++, vwt);
                *vcl_p++ = VEC_MADD(vu_m[stateChild][1].vx, wtdPartials, *vcl_p);
            }
           w += OFFSET*4;
           vcl_r += 2 * kExtraPatterns;
        }
    } else { // Integrate against a partial at the child

        const double* cl_q = gPartials[childIndex];
        V_Real * vcl_r = (V_Real *)cl_r;
        int v = 0;
        int w = 0;

        for(int l = 0; l < kCategoryCount; l++) {

            V_Real * vcl_p = (V_Real *)cl_p;

            VecUnion vu_m[OFFSET][2];
            AVX_PREFETCH_MATRIX(transMatrix + w, vu_m)

            for(int k = 0; k < kPatternCount; k++) {
                V_Real vclp_01, vclp_23;
                V_Real vwt = VEC_SPLAT(wt[l]);

                V_Real vcl_q0, vcl_q1, vcl_q2, vcl_q3;
                AVX_PREFETCH_PARTIALS(vcl_q,cl_q,v);

                vclp_01 = VEC_MULT(vcl_q0, vu_m[0][0].vx);
                vclp_01 = VEC_MADD(vcl_q1, vu_m[1][0].vx, vclp_01);
                vclp_01 = VEC_MADD(vcl_q2, vu_m[2][0].vx, vclp_01);
                vclp_01 = VEC_MADD(vcl_q3, vu_m[3][0].vx, vclp_01);
                vclp_23 = VEC_MULT(vcl_q0, vu_m[0][1].vx);
                vclp_23 = VEC_MADD(vcl_q1, vu_m[1][1].vx, vclp_23);
                vclp_23 = VEC_MADD(vcl_q2, vu_m[2][1].vx, vclp_23);
                vclp_23 = VEC_MADD(vcl_q3, vu_m[3][1].vx, vclp_23);
                vclp_01 = VEC_MULT(vclp_01, vwt);
                vclp_23 = VEC_MULT(vclp_23, vwt);

                *vcl_p++ = VEC_MADD(vclp_01, *vcl_r++, *vcl_p);
                *vcl_p++ = VEC_MADD(vclp_23, *vcl_r++, *vcl_p);

                v += 4;
            }
            w += 4*OFFSET;
            if (kExtraPatterns) {
                vcl_r += 2 * kExtraPatterns;
                v += 4 * kExtraPatterns;
            }

        }
    }

    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        double sumOverI = 0.0;
        for(int i = 0; i < kStateCount; i++) {
            sumOverI += freqs[i] * cl_p[u];
            u++;
        }

        outLogLikelihoodsTmp[k] = log(sumOverI);
    }


    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++)
            outLogLikelihoodsTmp[k] += scalingFactors[k];
    }

    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
    }

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;
        
    return returnCode;
}


BEAGLE_CPU_4_AVX_TEMPLATE
int BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::getPaddedPatternsModulus() {
	return 1;  // We currently do not vectorize across patterns
//	return 4;  // For single-precision, can operate on 4 patterns at a time
	// TODO Vectorize final log operations over patterns
}
    
BEAGLE_CPU_4_AVX_TEMPLATE
int BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::getPaddedPatternsModulus() {
//	return 2;  // For double-precision, can operate on 2 patterns at a time
	return 1;  // We currently do not vectorize across patterns
}

BEAGLE_CPU_4_AVX_TEMPLATE
const char* BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::getName() {
	return  getBeagleCPU4StateAVXName<float>();
}

BEAGLE_CPU_4_AVX_TEMPLATE
const char* BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::getName() {
    return  getBeagleCPU4StateAVXName<double>();
}

    
BEAGLE_CPU_4_AVX_TEMPLATE
const long BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT>::getFlags() {
	return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_THREADING_NONE |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_SINGLE |
            BEAGLE_FLAG_VECTOR_AVX;
}

BEAGLE_CPU_4_AVX_TEMPLATE
const long BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE>::getFlags() {
    return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_THREADING_NONE |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_DOUBLE |
            BEAGLE_FLAG_VECTOR_AVX;
}



///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPU4StateAVXImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
                                             int partialsBufferCount,
                                             int compactBufferCount,
                                             int stateCount,
                                             int patternCount,
                                             int eigenBufferCount,
                                             int matrixBufferCount,
                                             int categoryCount,
                                             int scaleBufferCount,
                                             int resourceNumber,
                                             int pluginResourceNumber,                                             
                                             long preferenceFlags,
                                             long requirementFlags,
                                             int* errorCode) {

    if (stateCount != 4) {
        return NULL;
    }

    BeagleCPU4StateAVXImpl<REALTYPE, T_PAD_4_AVX_DEFAULT, P_PAD_4_AVX_DEFAULT>* impl =
    		new BeagleCPU4StateAVXImpl<REALTYPE, T_PAD_4_AVX_DEFAULT, P_PAD_4_AVX_DEFAULT>();

    if (!CPUSupportsAVX()) {
        delete impl;
        return NULL;
    }

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber,  pluginResourceNumber, preferenceFlags, requirementFlags) == 0)
            return impl;
    }
    catch(...) {
        if (DEBUGGING_OUTPUT)
            std::cerr << "exception in initialize\n";
        delete impl;
        throw;
    }

    delete impl;

    return NULL;
}

BEAGLE_CPU_FACTORY_TEMPLATE
const char* BeagleCPU4StateAVXImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
	return getBeagleCPU4StateAVXName<BEAGLE_CPU_FACTORY_GENERIC>();
}

template <>
const long BeagleCPU4StateAVXImplFactory<double>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_AVX |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL|
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_FRAMEWORK_CPU;           
}

template <>
const long BeagleCPU4StateAVXImplFactory<float>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_AVX |
           BEAGLE_FLAG_PRECISION_SINGLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_FRAMEWORK_CPU;           
}


}
}

#endif //BEAGLE_CPU_4STATE_AVX_IMPL_HPP
