#pragma once

#include "gxnet.h"

#include <algorithm>

typedef struct tagCmdArgs {
	int mTrainingCount;
	int mEvalCount;
	int mEpochCount;
	int mMiniBatchCount;
	float mLearningRate;
	float mLambda;
	bool mIsDebug;
	bool mIsShuffle;
	const char * mModelPath;
} CmdArgs_t;


class GX_Utils {
public:
	template< class ForwardIt >
	static int max_index( ForwardIt first, ForwardIt last )
	{
		return std::distance( first, std::max_element( first, last ) );
	}

	static GX_DataType random();

	static GX_DataType calcSSE( const GX_DataVector & output, const GX_DataVector & target );

	static void printMnistImage( const char * tag, const GX_DataVector & data );

	static bool centerMnistImage( GX_DataVector & orgImage, GX_DataVector * newImage );

	static bool loadMnistImages( const int limitCount, const char * path, GX_DataMatrix * images );

	static bool loadMnistLabels( int limitCount, const char * path, GX_DataMatrix * labels, int maxClasses );

	static void printMatrix( const char * tag, const GX_DataMatrix & data,
			bool useSciFmt = true, bool colorMax = false);

	static void printVector( const char * tag, const GX_DataVector & data, bool useSciFmt = true );

	static void addMatrix( GX_DataMatrix * dest, const GX_DataMatrix & src );

	static bool save( const char * path, const GX_Network & network );

	static bool load( const char * path, GX_Network * network );

public:

	static void getCmdArgs( int argc, char * const argv[],
			const CmdArgs_t & defaultArgs, CmdArgs_t * args );
};

