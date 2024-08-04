
#include "gxutils.h"
#include "gxnet.h"
#include "gxact.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <regex>
#include <random>
#include <numeric>
#include <climits>
#include <algorithm>

#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include <arpa/inet.h>

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

void GX_Utils :: printMnistImage( const char * tag, const GX_DataVector & data )
{
	printf( "%s { %ld }\n", tag, data.size() );

	size_t len = sqrt( data.size() );

	for( size_t i = 0; i < len ; i++ ) {
		for( size_t j = 0; j < len; j++ ) {
			size_t idx = i * len + j;
			if( data[ idx ] != 0 ) {
				printf( "\e[1;31m1\e[0m " );
			} else {
				printf( "0 " );
			}
		}
		printf( "\n" );
	}
}

bool GX_Utils :: centerMnistImage( GX_DataVector & orgImage, GX_DataVector * newImage )
{
	bool ret = false;

	GX_DataType buff[ 28 ] [ 28 ];

	for( int x = 0; x < 28; x++ ) {
		for( int y = 0; y < 28; y++ ) {
			buff[ x ][ y ] = orgImage[ x * 28 + y ];
		}
	}

	int beginX = INT_MAX, beginY = INT_MAX, endX = INT_MIN, endY = INT_MIN;

	for( int x = 0; x < 28; x++ ) {
		for( int y = 0; y < 28; y++ ) {
			if( buff[ x ][ y ] != 0 ) {
				beginX = std::min( x, beginX );
				beginY = std::min( y, beginY );

				endX = std::max( x, endX );
				endY = std::max( y, endY );
			}
		}
	}

	endX++;
	endY++;

	int marginX = ( 28 - ( endX - beginX ) ) / 2;
	int marginY = ( 28 - ( endY - beginY ) ) / 2;

	if( marginX != beginX || marginY != beginY ) {

		GX_DataType newBuff[ 28 ][ 28 ];
		memset( newBuff, 0, sizeof( newBuff ) );

		newImage->resize( orgImage.size(), 0 );

		for( int x = beginX; x < endX; x++ ) {
			for( int y = beginY; y < endY; y++ ) {
				newBuff[ marginX + x - beginX ][ marginY + y - beginY ] = buff[ x ][ y ];
			}
		}

		for( int x = 0; x < 28; x++ ) {
			for( int y = 0; y < 28; y++ ) {
				( *newImage )[ x * 28 + y ] = newBuff[ x ][ y ];
			}
		}

		ret = true;
	}

	return ret;
}

bool GX_Utils :: expandMnistImage( GX_DataVector & orgImage, GX_DataVector * newImage )
{
	bool ret = true;

	newImage->resize( 32 * 32, 0 );

	for( int x = 0; x < 28; x++ ) {
		for( int y = 0; y < 28; y++ ) {
			( *newImage )[ ( x + 2 ) * 32 + y + 2 ] = orgImage[ x * 28 + y ];
		}
	}

	return ret;
}

bool GX_Utils :: loadMnistImages( const int limitCount, const char * path, GX_DataMatrix * images )
{
	std::ifstream file( path, std::ios::binary );

	if( ! file.is_open() ) {
		printf( "open %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	int magic = 0, rows = 0, cols = 0;
	int imageCount = 0, imageSize = 0;

	file.read( ( char * )&magic, sizeof( magic ) );
	magic = ntohl( magic );

	if( magic != 2051 ) return false;

	file.read( ( char * )&imageCount, sizeof( imageCount ) );
	imageCount = ntohl( imageCount );

	file.read( ( char * )&rows, sizeof( rows ) );
	rows = ntohl( rows );

	file.read( ( char * )&cols, sizeof( cols ));
	cols = ntohl( cols );

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

		images->emplace_back( GX_DataVector( imageSize ) );
		for( int j = 0; j < imageSize; j++ ) {
			images->back()[ j ] = buff[ j ] / 255.0;
		}
	}
	free( buff );

	printf( "%s load %s images %zu\n", __func__, path, images->size() );

	return ret;
}

bool GX_Utils :: loadMnistLabels( int limitCount, const char * path, GX_DataMatrix * labels, int maxClasses )
{
	std::ifstream file( path, std::ios::binary );

	if( ! file.is_open() ) {
		printf( "open %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	int magic = 0, labelCount = 0;

	file.read( ( char * )&magic, sizeof( magic ) );
	magic = ntohl( magic );

	if( magic != 2049 ) {
		printf( "read %s, invalid magic %d\n", path, magic );
		return false;
	}

	file.read( ( char * )&labelCount, sizeof( labelCount ) );
	labelCount = ntohl( labelCount );

	bool ret = true;

	if( limitCount > 0 ) labelCount = std::min( limitCount, labelCount );

	labels->reserve( labelCount );

	unsigned char buff = 0;

	for(int i = 0; i < labelCount; i++) {
		if( ( ! file.read( ( char * )&buff, 1 ) ) || buff >= maxClasses ) {
			printf( "%s read fail, label %d\n", __func__, buff );
			ret = false;
			break;
		}

		labels->emplace_back( GX_DataVector( maxClasses ) );
		labels->back()[ buff ] = 1;
	}

	printf( "%s load %s labels %zu\n", __func__, path, labels->size() );

	return ret;
}

void GX_Utils :: printMatrix( const char * tag, const GX_DataMatrix & data,
		bool useSciFmt, bool colorMax )
{
	printf( "%s { %ld }\n", tag, data.size() );
	for( size_t i = 0; i < data.size(); i++ ) {
		printf( "#%ld ", i );

		GX_DataType maxValue = *std::max_element( std::begin( data[ i ] ), std::end( data[ i ] ) );

		for( size_t j = 0; j < data[ i ].size(); j++ ) {
			const char * fmt = "%.2f ";

			if( colorMax && data[ i ][ j ] == maxValue ) {
				fmt = useSciFmt ? "\e[1;31m%8e\e[0m " : "\e[1;31m%.2f\e[0m ";
			} else {
				fmt = useSciFmt ? "%8e " : "%.2f ";
			}

			printf( fmt, data[ i ][ j ] );
		}

		printf( "\n" );
	}
}

void GX_Utils :: printVector( const char * tag, const GX_DataVector & data, const GX_Dims & dims, bool useSciFmt )
{
	GX_MDSpanRO ms( data, dims );

	printMDSpan( tag, ms, useSciFmt );
}

void GX_Utils :: printMDSpan( const char * tag, const GX_MDSpanRO & data, bool useSciFmt )
{
	GX_Dims dims( std::max( size_t(4), data.dims().size() ), 1 );

	std::copy( data.dims().rbegin(), data.dims().rend(), dims.rbegin() );

	printf( "{{{\n\n" );
	printf( "%s dims %zu { %zu, %zu, %zu, %zu }\n\n", tag, data.dims().size(), dims[ 0 ], dims[ 1 ], dims[ 2 ], dims[ 3 ] );

	for( size_t f = 0; f < dims[ 0 ]; f++ ) {
		if( dims[ 0 ] > 1 ) printf( "%s#%zu\n", tag, f );
		for( size_t c = 0; c < dims[ 1 ]; c++ ) {
			if( dims[ 1 ] > 1 ) printf( "%s#%zu\n", tag, c );
			for( size_t x = 0; x < dims[ 2 ]; x++ ) {
				for( size_t y = 0; y < dims[ 3 ]; y++ ) {
					GX_DataType value = 0;
					if( data.dims().size() == 1 ) value = data( y );
					if( data.dims().size() == 2 ) value = data( x, y );
					if( data.dims().size() == 3 ) value = data( c, x, y );
					if( data.dims().size() == 4 ) value = data( f, c, x, y );

					printf( useSciFmt ? "%.8e " : "%.2f ", value );
				}
				printf( "\n" );
			}
			printf( "\n" );
		}
		printf( "\n" );
	}
	printf( "}}}\n\n" );
}

void GX_Utils :: printVector( const char * tag, const GX_DataVector & data, bool useSciFmt )
{
	printf( "%s { %ld }\n", tag, data.size() );

	for( auto & i : data ) printf( useSciFmt ? "%.8e " : "%.2f ", i );

	printf( "\n" );
}

bool GX_Utils :: save( const char * path, const GX_Network & network )
{
	FILE * fp = fopen( path, "w" );

	if( NULL == fp ) return false;

	fprintf( fp, "Network: LayerCount = %ld; LossFuncType = %d;\n",
			network.getLayers().size(), network.getLossFuncType() );

	for( size_t i = 0; i < network.getLayers().size(); i++ ) {
		GX_BaseLayer * layer = network.getLayers() [ i ];

		fprintf( fp, "Layer#%ld: Type = %d; ActFuncType = %d; InputDims = %s;\n",
				i, layer->getType(),
				layer->getActFunc() ? layer->getActFunc()->getType() : -1,
				gx_vector2string( layer->getInputDims() ).c_str() );

		if( GX_BaseLayer::eMaxPool == layer->getType() ) {
			fprintf( fp, "Weights: PoolSize = %zu;\n", ((GX_MaxPoolLayer*)layer)->getPoolSize() );
		}
		if( GX_BaseLayer::eAvgPool == layer->getType() ) {
			fprintf( fp, "Weights: PoolSize = %zu;\n", ((GX_AvgPoolLayer*)layer)->getPoolSize() );
		}
		if( GX_BaseLayer::eConv == layer->getType() ) {
			GX_ConvLayer * conv = (GX_ConvLayer*)layer;
			fprintf( fp, "Weights: FilterDims = %s;\n", gx_vector2string( conv->getFilterDims() ).c_str() );
			fprintf( fp, "%s\n", gx_vector2string( conv->getFilters() ).c_str() );
			fprintf( fp, "Biases: Count = %zu;\n", conv->getBiases().size() );
			fprintf( fp, "%s\n", gx_vector2string( conv->getBiases() ).c_str() );
		}
		if( GX_BaseLayer::eFullConn == layer->getType() ) {
			GX_FullConnLayer * fc = (GX_FullConnLayer*)layer;
			fprintf( fp, "Weights: Count = %zu;\n", fc->getWeights().size() );
			for( size_t k = 0; k < fc->getWeights().size(); k++ ) {
				fprintf( fp, "%s\n", gx_vector2string( fc->getWeights()[ k ] ).c_str() );
			}
			fprintf( fp, "Biases: Count = %zu;\n", fc->getBiases().size() );
			fprintf( fp, "%s\n", gx_vector2string( fc->getBiases() ).c_str() );
		}
	}

	fclose( fp );

	return true;
}

bool GX_Utils :: load( const char * path, GX_Network * network )
{
	auto getString = []( std::string const & line, const char * fmt, const char * defaultValue ) {
		std::regex ex( fmt );
		std::smatch match;

		std::string value = defaultValue;
		if( std::regex_search( line, match, ex ) ) value = match[ match.size() - 1 ];

		return value;
	};

	std::ifstream fp( path );

	if( !fp ) return false;

	std::string line;

	// Network: LayerCount = x; LossFuncType = x;
	if( ! std::getline( fp, line ) ) return false;

	network->setLossFuncType( std::stoi( getString( line, "LossFuncType = (\\S+);", "1" ) ) );

	int layerCount = std::stoi( getString( line, "LayerCount = (\\S+);", "0" ) );

	network->getLayers().reserve( layerCount );

	for( int i = 0; i < layerCount; i++ ) {
		//Layer#x: Type = x; ActFuncType = x; InputDims = c,x,y;
		if( ! std::getline( fp, line ) ) return false;

		GX_BaseLayer * layer = NULL;

		int layerType = std::stoi( getString( line, "Type = (\\S+);", "0" ) );
		int actFuncType = std::stoi( getString( line, "ActFuncType = (\\S+);", "0" ) );

		GX_Dims inputDims;
	       	gx_string2vector( getString( line, "InputDims = (\\S+);", "0" ), &inputDims );

		if( GX_BaseLayer::eConv == layerType ) {
			// Weights: FilterDims = f,c,x,y;
			if( ! std::getline( fp, line ) ) return false;

			GX_Dims filterDims;
			gx_string2vector( getString( line, "FilterDims = (\\S+);", "0" ), &filterDims );

			if( ! std::getline( fp, line ) ) return false;

			GX_DataVector filters( gx_dims_flatten_size( filterDims ) );
			gx_string2valarray( line, &filters );

			// Biases: Count = xx;
			if( ! std::getline( fp, line ) ) return false;

			GX_DataVector biases( filterDims[ 0 ] );

			if( ! std::getline( fp, line ) ) return false;
			gx_string2valarray( line, &biases );

			layer = new GX_ConvLayer( inputDims, filters, filterDims, biases );
		}
		if( GX_BaseLayer::eMaxPool == layerType ) {
			// Weights: PoolSize = xx;
			if( ! std::getline( fp, line ) ) return false;

			int poolSize = std::stoi( getString( line, "PoolSize = (\\S+);", "0" ) );

			layer = new GX_MaxPoolLayer( inputDims, poolSize );
		}
		if( GX_BaseLayer::eAvgPool == layerType ) {
			// Weights: PoolSize = xx;
			if( ! std::getline( fp, line ) ) return false;

			int poolSize = std::stoi( getString( line, "PoolSize = (\\S+);", "0" ) );

			layer = new GX_AvgPoolLayer( inputDims, poolSize );
		}
		if( GX_BaseLayer::eFullConn == layerType ) {
			// Weights: Count = xx;
			if( ! std::getline( fp, line ) ) return false;

			int count = std::stoi( getString( line, "Count = (\\S+);", "0" ) );

			layer = new GX_FullConnLayer( count, gx_dims_flatten_size( inputDims ) );

			GX_DataMatrix weights( count );
			for( int i = 0; i < count; i++ ) {
				if( ! std::getline( fp, line ) ) return false;

				weights[ i ].resize( gx_dims_flatten_size( inputDims ) );
				gx_string2valarray( line, &( weights[ i ] ) );
			}

			// Biases: Count = xx;
			if( ! std::getline( fp, line ) ) return false;

			GX_DataVector biases( count );

			if( ! std::getline( fp, line ) ) return false;
			gx_string2valarray( line, &biases );

			((GX_FullConnLayer*)layer)->setWeights( weights, biases );
		}

		if( actFuncType > 0 ) layer->setActFunc( new GX_ActFunc( actFuncType ) );

		network->addLayer( layer );
	}

	return true;
}

void GX_Utils :: getCmdArgs( int argc, char * const argv[],
		const CmdArgs_t & defaultArgs, CmdArgs_t * args )
{
	static struct option opts[] = {
		{ "model",     required_argument,  NULL, 1 },
		{ "training",  required_argument,  NULL, 2 },
		{ "eval",      required_argument,  NULL, 3 },
		{ "epoch",     required_argument,  NULL, 4 },
		{ "minibatch", required_argument,  NULL, 5 },
		{ "lr",        required_argument,  NULL, 6 },
		{ "lambda",    required_argument,  NULL, 7 },
		{ "shuffle",   required_argument,  NULL, 8 },
		{ "debug",     no_argument,        NULL, 9 },
		{ "help",      no_argument,        NULL, 10 },
		{ 0, 0, 0, 0}
	};

	extern char *optarg ;
	int c ;

	*args = defaultArgs;;

	while( ( c = getopt_long( argc, argv, "v", opts, NULL )) != EOF ) {
		switch ( c ) {
			case 1:
				args->mModelPath = optarg;
				break;
			case 2:
				args->mTrainingCount = atoi( optarg );
				break;
			case 3:
				args->mEvalCount = atoi( optarg );
				break;
			case 4:
				args->mEpochCount = atoi( optarg );
				break;
			case 5:
				args->mMiniBatchCount = atoi( optarg );
				break;;
			case 6:
				args->mLearningRate = std::stof( optarg );
				break;
			case 7:
				args->mLambda = std::stof( optarg );
				break;
			case 8:
				args->mIsShuffle = 0 == atoi( optarg ) ? false : true;
				break;
			case 9:
				args->mIsDebug = true;
				break;
			case '?' :
			case 'v' :
				printf( "Usage: %s [-v]\n", argv[ 0 ] );
				printf( "\t--model <model path> if path exist, then continue training\n" );
				printf( "\t--training <training data count> 0 for all, default is %d\n", defaultArgs.mTrainingCount );
				printf( "\t--eval <eval count> 0 for all, default is %d\n", defaultArgs.mEvalCount );
				printf( "\t--epoch <epoch count> default is %d\n", defaultArgs.mEpochCount );
				printf( "\t--minibatch <mini batch count> default is %d\n", defaultArgs.mMiniBatchCount );
				printf( "\t--lr <learning rate> default is %.2f\n", defaultArgs.mLearningRate );
				printf( "\t--lambda <lambda> default is %.2f\n", defaultArgs.mLambda );
				printf( "\t--shuffle <shuffle> 0 for no shuffle, otherwise shuffle, default is %d\n", defaultArgs.mIsShuffle );
				printf( "\t--debug debug mode on\n" );
				printf( "\t--help show usage\n" );
				exit( 0 );
		}
	}

	printf( "args:\n" );
	printf( "\ttrainingCount %d, evalCount %d\n", args->mTrainingCount, args->mEvalCount );
	printf( "\tepochCount %d, miniBatchCount %d, learningRate %f, lambda %f\n",
		args->mEpochCount, args->mMiniBatchCount, args->mLearningRate, args->mLambda );
	printf( "\tshuffle %s, debug %s\n", args->mIsShuffle ? "true" : "false", args->mIsDebug ? "true" : "false" );
	printf( "\tmodelPath %s\n", NULL == args->mModelPath ? "NULL" : args->mModelPath );
	printf( "\n" );
}

