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
#include <stdio.h>

void check( const char * tag, GX_Network & network, GX_DataMatrix & input, GX_DataMatrix & target, bool isDebug )
{
	printf( "%s( %s, ..., input { %ld }, target { %ld } )\n", __func__, tag, input.size(), target.size() );

	if( isDebug ) network.print();

	GX_DataMatrix confusionMatrix;
	GX_DataVector targetTotal;

	size_t maxClasses = target[ 0 ].size();
	confusionMatrix.resize( maxClasses );
	targetTotal.resize( maxClasses );
	for( size_t i = 0; i < maxClasses; i++ ) confusionMatrix[ i ].resize( maxClasses, 0.0 );

	int correct = 0;

	for( size_t i = 0; i < input.size(); i++ ) {

		GX_DataMatrix output;

		bool ret = network.forward( input[ i ], &output );

		if( ! ret ) {
			printf( "forward fail\n" );
			return;
		}

		int outputType = GX_Utils::max_index( output.back().begin(), output.back().end() );
		int targetType = GX_Utils::max_index( target[ i ].begin(), target[ i ].end() );

		if( isDebug ) printf( "forward %d, index %zu, %d %d\n", ret, i, outputType, targetType );

		if( outputType == targetType ) correct++;

		confusionMatrix[ targetType ][ outputType ] += 1;
		targetTotal[ targetType ] += 1;

		for( size_t j = 0; isDebug && j < output.back().size() && j < 10; j++ ) {
			printf( "\t%zu %.8f %.8f\n", j, output.back()[ j ], target[ i ][ j ] );
		}
	}

	printf( "check %s, %d/%ld = %.2f\n", tag, correct, input.size(), ((float)correct) / input.size() );

	for( size_t i = 0; i < confusionMatrix.size(); i++ ) {
		for( auto & item : confusionMatrix[ i ] ) item = item / targetTotal[ i ];
	}

	GX_Utils::printMatrix( "confusion matrix", confusionMatrix, false, true );
}

bool loadData( const CmdArgs_t & args, GX_DataMatrix * input, GX_DataMatrix * target,
		GX_DataMatrix * input4eval, GX_DataMatrix * target4eval )
{
	const char * path = "mnist/train-images.idx3-ubyte";
	if( ! GX_Utils::loadMnistImages( args.mTrainingCount, path, input ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/train-labels.idx1-ubyte";
	if( ! GX_Utils::loadMnistLabels( args.mTrainingCount, path, target, 10 ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	// load rotated images
	path = "mnist/train-images.idx3-ubyte.rot";
	if( 0 == access( path, F_OK ) ) {
		if( ! GX_Utils::loadMnistImages( args.mTrainingCount, path, input ) ) {
			printf( "read %s fail\n", path );
			return false;
		}

		path = "mnist/train-labels.idx1-ubyte.rot";
		if( ! GX_Utils::loadMnistLabels( args.mTrainingCount, path, target, 10 ) ) {
			printf( "read %s fail\n", path );
			return false;
		}
	}

	// center mnist images
	size_t orgSize = input->size();

	for( size_t i = 0; i < orgSize; i++ ) {
		GX_DataVector newImage;
		if( GX_Utils::centerMnistImage( ( *input )[ i ], &newImage ) ) {
			input->push_back( newImage );
			target->push_back( ( *target )[ i ] );
		}
	}

	path = "mnist/t10k-images.idx3-ubyte";
	if( ! GX_Utils::loadMnistImages( args.mEvalCount, path, input4eval ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/t10k-labels.idx1-ubyte";
	if( ! GX_Utils::loadMnistLabels( args.mEvalCount, path, target4eval, 10 ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	printf( "input { %zu }, target { %zu }, input4eval { %zu }, target4eval { %zu }\n",
			input->size(), target->size(), input4eval->size(), target4eval->size() );

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

		//network.setLossFuncType( GX_Network::eCrossEntropy );

		if( NULL != args.mModelPath && 0 == access( args.mModelPath, F_OK ) ) {
			if(  GX_Utils::load( args.mModelPath, &network ) ) {
				printf( "continue training %s\n", args.mModelPath );
			} else {
				printf( "load( %s ) fail\n", args.mModelPath );
				return;
			}
		} else {
			network.addLayer( 30, input[ 0 ].size() );
			network.addLayer( target[ 0 ].size(), 30 );
		}

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
		.mEpochCount = 5,
		.mMiniBatchCount = 100,
		.mLearningRate = 3.0,
		.mLambda = 5.0,
		.mIsDebug = false,
		.mIsShuffle = true,
	};

	CmdArgs_t args = defaultArgs;

	GX_Utils::getCmdArgs( argc, argv, defaultArgs, &args );

	test( args );

	return 0;
}

