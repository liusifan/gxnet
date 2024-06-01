#include "gxnet.h"
#include "gxutils.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <set>
#include <float.h>

#include <unistd.h>
#include <syslog.h>

void check( const char * tag, GX_Network & network, GX_DataMatrix & input, GX_DataMatrix & target, bool isDebug )
{
	printf( "%s( %s, ..., input { %ld }, target { %ld } )\n", __func__, tag, input.size(), target.size() );

	if( isDebug ) network.print();

	int correct = 0;

	for( size_t i = 0; i < input.size(); i++ ) {

		GX_DataMatrix output;

		bool ret = network.forward( input[ i ], &output );

		if( ! ret ) {
			printf( "forward fail\n" );
			return;
		}

		int outputType = std::max_element( output.back().begin(), output.back().end() ) - output.back().begin();
		int targetType = std::max_element( target[ i ].begin(), target[ i ].end() ) - target[ i ].begin();

		if( isDebug ) printf( "forward %d, index %zu, %d %d\n", ret, i, outputType, targetType );

		if( outputType == targetType ) correct++;

		for( size_t j = 0; isDebug && j < output.back().size() && j < 10; j++ ) {
			printf( "\t%zu %.8f %.8f\n", j, output.back()[ j ], target[ i ][ j ] );
		}
	}

	printf( "GX_Network %s, %d/%ld = %.2f\n", tag, correct, input.size(), ((float)correct) / input.size() );
}

bool loadData( const CmdArgs_t & args, GX_DataMatrix * input, GX_DataMatrix * target,
		GX_DataMatrix * input4eval, GX_DataMatrix * target4eval )
{
	const char * path = "mnist/train-images.idx3-ubyte";
	if( ! GX_Utils::readMnistImages( args.mTrainingCount, path, input ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/train-labels.idx1-ubyte";
	if( ! GX_Utils::readMnistLabels( args.mTrainingCount, path, target ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/t10k-images.idx3-ubyte";
	if( ! GX_Utils::readMnistImages( args.mEvalCount, path, input4eval ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/t10k-labels.idx1-ubyte";
	if( ! GX_Utils::readMnistLabels( args.mEvalCount, path, target4eval ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	return true;
}

void test( const CmdArgs_t & args )
{
	GX_DataMatrix input, target, input4eval, target4eval;

	if( ! loadData( args, &input, &target, &input4eval, &target4eval ) ) {
		printf( "loadData fail\n" );
		return;
	}

	const char * path = "./mnist.model";
	//train & save model
	{

		GX_Network network;

		network.addLayer( 30, input[ 0 ].size() );
		network.addLayer( target[ 0 ].size(), 30 );

		check( "before train", network, input4eval, target4eval, args.mIsDebug );

		bool ret = network.train( input, target, args.mIsShuffle,
				args.mEpochCount, args.mMiniBatchCount, args.mLearningRate, args.mLambda );

		GX_Utils::save( path, network );

		printf( "train %s\n", ret ? "succ" : "fail" );

		check( "after train", network, input4eval, target4eval, args.mIsDebug );
	}

	//load model
	{
		GX_Network network;

		GX_Utils::load( path, &network );

		check( "load model", network, input4eval, target4eval, args.mIsDebug );
	}
}

int main( const int argc, char * argv[] )
{
	CmdArgs_t defaultArgs = {
		.mTrainingCount = 0,
		.mEvalCount = 0,
		.mIsShuffle = true,
		.mEpochCount = 5,
		.mMiniBatchCount = 10,
		.mLearningRate = 3.0,
		.mLambda = 5.0,
		.mIsDebug = false,
	};

	CmdArgs_t args = defaultArgs;

	GX_Utils::getCmdArgs( argc, argv, defaultArgs, &args );

	test( args );

	return 0;
}

