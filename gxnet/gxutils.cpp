
#include "gxutils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <regex>
#include <random>
#include <numeric>

#include <unistd.h>
#include <assert.h>

GX_DataType GX_Utils :: calcSSE( const GX_DataVector & output, const GX_DataVector & target )
{
	assert( output.size() == target.size() );

	GX_DataType sse = 0;
	for( size_t x = 0; x < target.size(); x++ ) {
		GX_DataType tmp = target[ x ] - output[ x ];
		sse += tmp * tmp;
	}

	return sse;
}

GX_DataType GX_Utils :: random()
{
	static std::random_device rd;
	static std::mt19937 gen( rd() );
 
	static std::normal_distribution<> dist( 0, 1 );
 
	return dist( gen );
}

void GX_Utils :: addMatrix( GX_DataMatrix * dest, const GX_DataMatrix & src )
{
	dest->reserve( src.size() );
	for( size_t i = 0; i < src.size(); i++ ) {
		if( dest->size() <= i ) {
			dest->push_back( GX_DataVector() );
			dest->back().resize( src[ i ].size(), 0 );
		}

		for( size_t j = 0; j < src[ i ].size(); j++ ) {
			( *dest )[ i ][ j ] += src[ i ][ j ];
		}
	}
}

void GX_Utils :: printMnistImage( const char * tag, const GX_DataVector & data )
{
	printf( "%s { %ld }\n", tag, data.size() );

	for( int i = 0; i < 28; i++ ) {
		for( int j = 0; j < 28; j++ ) {
			int idx = i * 28 + j;
			printf( "%s ",  idx < data.size() && data[ idx ] != 0 ? "1" : "0" );
		}
		printf( "\n" );
	}
}

bool GX_Utils :: readMnistImages( const int limitCount, const char * path, GX_DataMatrix * images )
{
	auto reverseInt = []( int i ) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	std::ifstream file( path, std::ios::binary );

	if( ! file.is_open() ) {
		printf( "open %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	int magic = 0, rows = 0, cols = 0;
	int imageCount = 0, imageSize = 0;

	file.read( ( char * )&magic, sizeof( magic ) );
	magic = reverseInt( magic );

	if( magic != 2051 ) return false;

	file.read( ( char * )&imageCount, sizeof( imageCount ) );
	imageCount = reverseInt( imageCount );

	file.read( ( char * )&rows, sizeof( rows ) );
	rows = reverseInt( rows );

	file.read( ( char * )&cols, sizeof( cols ));
	cols = reverseInt( cols );

	imageSize = rows * cols;

	bool ret = true;

	if( limitCount > 0 ) imageCount = std::min( limitCount, imageCount );

	images->reserve( imageCount );

	unsigned char * buff = ( unsigned char * )malloc( imageSize );
	for( int i = 0; i < imageCount; i++ ) {
		if( ! file.read( (char*)buff, imageSize ) ) {
			printf( "%s read fail\n", __func__ );
			ret = false;
			break;
		}

		images->push_back( GX_DataVector() );
		images->back().reserve( imageSize );
		for( int j = 0; j < imageSize; j++ ) {
			images->back().push_back( buff[ j ] / 255.0 );
		}
	}
	free( buff );

	return ret;
}

bool GX_Utils :: readMnistLabels( int limitCount, const char * path, GX_DataMatrix * labels )
{
	auto reverseInt = []( int i ) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	std::ifstream file( path, std::ios::binary );

	if( ! file.is_open() ) {
		printf( "open %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	int magic = 0, labelCount = 0;

	file.read( ( char * )&magic, sizeof( magic ) );
	magic = reverseInt( magic );

	if(magic != 2049) return false;

	file.read( ( char * )&labelCount, sizeof( labelCount ) );
	labelCount = reverseInt( labelCount );

	bool ret = true;

	if( limitCount > 0 ) labelCount = std::min( limitCount, labelCount );

	labels->reserve( labelCount );

	unsigned char buff = 0;

	for(int i = 0; i < labelCount; i++) {
		if( ( ! file.read( ( char * )&buff, 1 ) ) || buff >= 10 ) {
			printf( "%s read fail\n", __func__ );
			ret = false;
			break;
		}

		labels->push_back( GX_DataVector() );
		labels->back().resize( 10, 0 );
		labels->back()[ buff ] = 1;
	}

	return ret;
}

void GX_Utils :: printMatrix( const char * tag, const GX_DataMatrix & data )
{
	printf( "%s { %ld }\n", tag, data.size() );
	for( size_t i = 0; i < data.size(); i++ ) {
		printf( "#%ld ", i );
		for( auto & j : data[ i ] ) printf( "%.8e ", j );
		printf( "\n" );
	}
}

void GX_Utils :: printVector( const char * tag, const GX_DataVector & data )
{
	printf( "%s { %ld }\n", tag, data.size() );

	for( auto & i : data ) printf( "%.8e ", i );

	printf( "\n" );
}

bool GX_Utils :: save( const char * path, const GX_Network & network )
{
	FILE * fp = fopen( path, "w" );

	if( NULL == fp ) return false;

	fprintf( fp, "Layers: %ld\n", network.getLayers().size() );

	for( size_t i = 0; i < network.getLayers().size(); i++ ) {
		const GX_NeuronPtrVector & neurons = network.getLayers()[ i ]->getNeurons();

		fprintf( fp, "Layer#%ld: %ld\n", i, neurons.size() );

		for( size_t j = 0; j < neurons.size(); j++ ) {
			GX_Neuron * neuron = neurons[ j ];
			const GX_DataVector & weights = neuron->getWeights();

			fprintf( fp, "Neuron#%ld: %ld\n", j, weights.size() );
			fprintf( fp, "Bias#%ld: %e\n", j, neurons[ j ]->getBias() );

			fprintf( fp, "Weights#%ld:\n\t", j );

			for( size_t k = 0; k < weights.size(); k++ ) {
				fprintf( fp, "%s%e", 0 == k ? "" : ", ", weights[ k ] );
			}
			fprintf( fp, "\n" );
		}
	}

	fclose( fp );

	return true;
}

bool GX_Utils :: load( const char * path, GX_Network * network )
{
	auto getNumber = []( std::string const & line ) {
		const std::regex colon( ":" );

		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), colon, -1 ),
			std::sregex_token_iterator() };

		return srow.size() > 0 ? std::stod( srow[ 1 ] ) : 0;
	};

	auto getNumberVector = []( std::string const & line, GX_DataVector * data ) {
		const std::regex colon( "," );

		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), colon, -1 ),
			std::sregex_token_iterator() };

		data->reserve( srow.size() );
		for( size_t i = 0; i < srow.size(); i++ ) {
			( *data )[ i ] = std::stod( srow[ i ] );
		}
	};

	const std::regex colon( ":" ), comma( "," );

	std::ifstream fp( path );

	if( !fp ) return false;

	std::string line;

	// Layers: xxx
	if( ! std::getline( fp, line ) ) return false;

	int layerCount = getNumber( line );

	network->getLayers().reserve( layerCount );

	for( int i = 0; i < layerCount; i++ ) {
		// Layer#x: xxx
		if( ! std::getline( fp, line ) ) return false;

		int neuronCount = getNumber( line );

		GX_Layer * layer = NULL;

		for( int j = 0; j < neuronCount; j++ ) {

			// Neuron#x: xxx
			if( ! std::getline( fp, line ) ) return false;

			int weightCount = getNumber( line );

			if( NULL == layer )  layer = new GX_Layer( neuronCount, weightCount );

			GX_Neuron * neuron = layer->getNeurons()[ j ];

			// Bias#x: xxx
			if( ! std::getline( fp, line ) ) return false;

			neuron->setBias( getNumber( line ) );

			// Weights#x: xxx:
			if( ! std::getline( fp, line ) ) return false;

			// xxx, xxx, xxx
			if( ! std::getline( fp, line ) ) return false;

			getNumberVector( line, &neuron->getWeights() );
		}

		network->getLayers().push_back( layer );
	}

	return true;
}

void GX_Utils :: getCmdArgs( int argc, char * const argv[],
		const CmdArgs_t & defaultArgs, CmdArgs_t * args )
{
	extern char *optarg ;
	int c ;

	*args = defaultArgs;;

	while( ( c = getopt ( argc, argv, "a:t:c:e:l:n:dv" )) != EOF ) {
		switch ( c ) {
			case 't':
				args->mTrainingCount = atoi( optarg );
				break;
			case 'c':
				args->mEvalCount = atoi( optarg );
				break;
			case 'e' :
				args->mEpochCount = atoi( optarg );
				break;
			case 'b':
				args->mMiniBatchCount = atoi( optarg );
				break;;
			case 'l':
				args->mLearningRate = std::stof( optarg );
				break;
			case 'a':
				args->mLambda = std::stof( optarg );
				break;
			case 's':
				args->mIsShuffle = 0 == atoi( optarg ) ? false : true;
				break;
			case 'd':
				args->mIsDebug = true;
				break;
			case '?' :
			case 'v' :
				printf( "Usage: %s [-v]\n", argv[ 0 ] );
				printf( "\t-t <training data count> 0 for all, default is %d\n", defaultArgs.mTrainingCount );
				printf( "\t-c <eval count> 0 for all, default is %d\n", defaultArgs.mEvalCount );
				printf( "\t-e <epoch count> default is %d\n", defaultArgs.mEpochCount );
				printf( "\t-b <mini batch count> default is %d\n", defaultArgs.mMiniBatchCount );
				printf( "\t-l <learning rate> default is %.2f\n", defaultArgs.mLearningRate );
				printf( "\t-a <lambda> default is %.2f\n", defaultArgs.mLambda );
				printf( "\t-s <shuffle> 0 for no shuffle, otherwise shuffle, default is %d\n", defaultArgs.mIsShuffle );
				printf( "\t-d debug mode on\n" );
				printf( "\t-v show usage\n" );
				exit( 0 );
		}
	}

	printf( "args:\n" );
	printf( "\ttrainingCount %d, evalCount %d\n", args->mTrainingCount, args->mEvalCount );
	printf( "\tepochCount %d, miniBatchCount %d, learningRate %f, lambda %f\n",
		args->mEpochCount, args->mMiniBatchCount, args->mLearningRate, args->mLambda );
	printf( "\tisShuffle %s, isDebug %s\n", args->mIsShuffle ? "true" : "false", args->mIsDebug ? "true" : "false" );
}

