#pragma once

#include "gxnet.h"

typedef struct tagCmdArgs {
	int mTrainingCount;
	int mEvalCount;
	int mEpochCount;
	int mMiniBatchCount;
	float mLearningRate;
	float mLambda;
	bool mIsDebug;
	bool mIsShuffle;
} CmdArgs_t;


class GX_Utils {
public:
	static GX_DataType random();

	static GX_DataType calcSSE( const GX_DataVector & output, const GX_DataVector & target );

	static void printMnistImage( const char * tag, const GX_DataVector & data );

	static bool readMnistImages( const int limitCount, const char * path, GX_DataMatrix * images );

	static bool readMnistLabels( int limitCount, const char * path, GX_DataMatrix * labels );

	static void printMatrix( const char * tag, const GX_DataMatrix & data );

	static void printVector( const char * tag, const GX_DataVector & data );

	static void addMatrix( GX_DataMatrix * dest, const GX_DataMatrix & src );

	static bool save( const char * path, const GX_Network & network );

	static bool load( const char * path, GX_Network * network );

public:

	static void getCmdArgs( int argc, char * const argv[],
			const CmdArgs_t & defaultArgs, CmdArgs_t * args );
};

