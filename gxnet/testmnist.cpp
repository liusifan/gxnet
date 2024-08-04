#include "gxnet.h"
#include "gxact.h"
#include "gxutils.h"
#include "gxeval.h"

#include <unistd.h>

bool loadData( const CmdArgs_t & args, GX_DataMatrix * input, GX_DataMatrix * target,
		GX_DataMatrix * input4eval, GX_DataMatrix * target4eval )
{
	const char * path = "mnist/train-images-idx3-ubyte";
	if( ! GX_Utils::loadMnistImages( args.mTrainingCount, path, input ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/train-labels-idx1-ubyte";
	if( ! GX_Utils::loadMnistLabels( args.mTrainingCount, path, target, 10 ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	// load rotated images
	path = "mnist/train-images-idx3-ubyte.rot";
	if( 0 == access( path, F_OK ) ) {
		if( ! GX_Utils::loadMnistImages( args.mTrainingCount, path, input ) ) {
			printf( "read %s fail\n", path );
			return false;
		}

		path = "mnist/train-labels-idx1-ubyte.rot";
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
			input->emplace_back( newImage );
			target->emplace_back( ( *target )[ i ] );
		}
	}

	path = "mnist/t10k-images-idx3-ubyte";
	if( ! GX_Utils::loadMnistImages( args.mEvalCount, path, input4eval ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "mnist/t10k-labels-idx1-ubyte";
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

		network.setLossFuncType( GX_Network::eCrossEntropy );

		if( NULL != args.mModelPath && 0 == access( args.mModelPath, F_OK ) ) {
			if(  GX_Utils::load( args.mModelPath, &network ) ) {
				printf( "continue training %s\n", args.mModelPath );
			} else {
				printf( "load( %s ) fail\n", args.mModelPath );
				return;
			}
		} else {
			GX_BaseLayer * layer = NULL;

			layer = new GX_FullConnLayer( 30, input[ 0 ].size() );
			layer->setActFunc( GX_ActFunc::sigmoid() );
			network.addLayer( layer );

			layer = new GX_FullConnLayer( target[ 0 ].size(), layer->getOutputSize() );
			layer->setActFunc( GX_ActFunc::softmax() );
			network.addLayer( layer );
		}

		gx_eval( "before train", network, input4eval, target4eval, args.mIsDebug );

		network.print();

		bool ret = network.train( input, target,
				args.mEpochCount, args.mMiniBatchCount, args.mLearningRate, args.mLambda );

		GX_Utils::save( path, network );

		printf( "train %s\n", ret ? "succ" : "fail" );

		//gx_eval( "after train", network, input4eval, target4eval, args.mIsDebug );
	}

	//load model
	{
		GX_Network network;

		GX_Utils::load( path, &network );

		gx_eval( "load model", network, input4eval, target4eval, args.mIsDebug );
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

