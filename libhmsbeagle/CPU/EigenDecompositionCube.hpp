/*
 * EigenDecompositionCube.cpp
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */
#ifndef _EigenDecompositionCube_hpp_
#define _EigenDecompositionCube_hpp_

#include "libhmsbeagle/CPU/EigenDecompositionCube.h"


namespace beagle {
namespace cpu {

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

template <typename REALTYPE>
EigenDecompositionCube<REALTYPE>::EigenDecompositionCube(int decompositionCount,
											         int stateCount,
											         int categoryCount)
											         : EigenDecomposition<REALTYPE>(decompositionCount,
																				stateCount,
																				categoryCount) {
    gEigenValues = (REALTYPE**) malloc(sizeof(REALTYPE*) * kEigenDecompCount);
    if (gEigenValues == NULL)
        throw std::bad_alloc();
    
    gCMatrices = (REALTYPE**) malloc(sizeof(REALTYPE*) * kEigenDecompCount);
    if (gCMatrices == NULL)
    	throw std::bad_alloc();
    
    for (int i = 0; i < kEigenDecompCount; i++) {    	
    	gCMatrices[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount * kStateCount * kStateCount);
    	if (gCMatrices[i] == NULL)
    		throw std::bad_alloc();
    
    	gEigenValues[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount);
    	if (gEigenValues[i] == NULL)
    		throw std::bad_alloc();
    }
    
    matrixTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount);
    firstDerivTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount);
    secondDerivTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount);
}

template <typename REALTYPE>
EigenDecompositionCube<REALTYPE>::~EigenDecompositionCube() {
	
	for(int i=0; i<kEigenDecompCount; i++) {
		free(gCMatrices[i]);
		free(gEigenValues[i]);
	}
	free(gCMatrices);
	free(gEigenValues);
	free(matrixTmp);
}

template <typename REALTYPE>
void EigenDecompositionCube<REALTYPE>::setEigenDecomposition(int eigenIndex,
										           const double* inEigenVectors,
                                                   const double* inInverseEigenVectors,
                                                   const double* inEigenValues) {
		
    int l = 0;
    for (int i = 0; i < kStateCount; i++) {
        gEigenValues[eigenIndex][i] = inEigenValues[i];
        for (int j = 0; j < kStateCount; j++) {
            for (int k = 0; k < kStateCount; k++) {
                gCMatrices[eigenIndex][l] = inEigenVectors[(i * kStateCount) + k]
                        * inInverseEigenVectors[(k * kStateCount) + j];
                l++;
            }
        }
    }
}

template <typename REALTYPE>
void EigenDecompositionCube<REALTYPE>::updateTransitionMatrices(int eigenIndex,
                                                      const int* probabilityIndices,
                                                      const int* firstDerivativeIndices,
                                                      const int* secondDerivativeIndices,
                                                      const double* edgeLengths,
                                                      const double* categoryRates,
                                                      REALTYPE** transitionMatrices,
                                                      int count) {
	if (firstDerivativeIndices != NULL && secondDerivativeIndices != NULL) {
		for (int u = 0; u < count; u++) {
			REALTYPE* transitionMat = transitionMatrices[probabilityIndices[u]];
			REALTYPE* firstDerivMat = transitionMatrices[firstDerivativeIndices[u]];
			REALTYPE* secondDerivMat = transitionMatrices[secondDerivativeIndices[u]];
			int n = 0;
			for (int l = 0; l < kCategoryCount; l++) {
				
				for (int i = 0; i < kStateCount; i++) {
					REALTYPE scaledEigenValue = gEigenValues[eigenIndex][i] * ((REALTYPE)categoryRates[l]);
					matrixTmp[i] = exp(scaledEigenValue * ((REALTYPE)edgeLengths[u]));
					firstDerivTmp[i] = scaledEigenValue * matrixTmp[i];
					secondDerivTmp[i] = scaledEigenValue * firstDerivTmp[i];
				}
				
				int m = 0;
				for (int i = 0; i < kStateCount; i++) {
					for (int j = 0; j < kStateCount; j++) {
						REALTYPE sum = 0.0;
						REALTYPE sumD1 = 0.0;
						REALTYPE sumD2 = 0.0;
						for (int k = 0; k < kStateCount; k++) {
							sum += gCMatrices[eigenIndex][m] * matrixTmp[k];
							sumD1 += gCMatrices[eigenIndex][m] * firstDerivTmp[k];
							sumD2 += gCMatrices[eigenIndex][m] * secondDerivTmp[k];
							m++;
						}
						if (sum > 0)
							transitionMat[n] = sum;
						else
							transitionMat[n] = 0;
						firstDerivMat[n] = sumD1;
						secondDerivMat[n] = sumD2;
						n++;
					}
#ifdef PAD_MATRICES
					transitionMat[n] = 1.0;
					n++;
#endif
				}
			}
		}
	} else if (firstDerivativeIndices != NULL) {
		for (int u = 0; u < count; u++) {
			REALTYPE* transitionMat = transitionMatrices[probabilityIndices[u]];
			REALTYPE* firstDerivMat = transitionMatrices[firstDerivativeIndices[u]];
			int n = 0;
			for (int l = 0; l < kCategoryCount; l++) {
				
				for (int i = 0; i < kStateCount; i++) {
					REALTYPE scaledEigenValue = gEigenValues[eigenIndex][i] * ((REALTYPE)categoryRates[l]);
					matrixTmp[i] = exp(scaledEigenValue * ((REALTYPE)edgeLengths[u]));
					firstDerivTmp[i] = scaledEigenValue * matrixTmp[i];
				}
				
				int m = 0;
				for (int i = 0; i < kStateCount; i++) {
					for (int j = 0; j < kStateCount; j++) {
						REALTYPE sum = 0.0;
						REALTYPE sumD1 = 0.0;
						REALTYPE sumD2 = 0.0;
						for (int k = 0; k < kStateCount; k++) {
							sum += gCMatrices[eigenIndex][m] * matrixTmp[k];
							sumD1 += gCMatrices[eigenIndex][m] * firstDerivTmp[k];
							m++;
						}
						if (sum > 0)
							transitionMat[n] = sum;
						else
							transitionMat[n] = 0;
						firstDerivMat[n] = sumD1;
						n++;
					}
#ifdef PAD_MATRICES
					transitionMat[n] = 1.0;
					n++;
#endif
				}
			}
		}
	} else {		
		for (int u = 0; u < count; u++) {
			REALTYPE* transitionMat = transitionMatrices[probabilityIndices[u]];
			int n = 0;
			for (int l = 0; l < kCategoryCount; l++) {
				
                for (int i = 0; i < kStateCount; i++) {
					matrixTmp[i] = exp(gEigenValues[eigenIndex][i] * ((REALTYPE)edgeLengths[u] * categoryRates[l]));
                }
				
				int m = 0;
				for (int i = 0; i < kStateCount; i++) {
					for (int j = 0; j < kStateCount; j++) {
						REALTYPE sum = 0.0;
						for (int k = 0; k < kStateCount; k++) {
							sum += gCMatrices[eigenIndex][m] * matrixTmp[k];
							m++;
						}
						if (sum > 0)
							transitionMat[n] = sum;
						else
							transitionMat[n] = 0;
						n++;
					}
#ifdef PAD_MATRICES
					transitionMat[n] = 1.0;
					n++;
#endif
				}
			}
			
			if (DEBUGGING_OUTPUT) {
                int kMatrixSize = kStateCount * kStateCount;
				fprintf(stderr,"transitionMat index=%d brlen=%.5f\n", probabilityIndices[u], edgeLengths[u]);
				for ( int w = 0; w < (20 > kMatrixSize ? 20 : kMatrixSize); ++w)
					fprintf(stderr,"transitionMat[%d] = %.5f\n", w, transitionMat[w]);
			}
		}
	}
}

} // cpu
} // beagle

#endif // _EigenDecompositionCube_hpp_

