
#include "gxutils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <regex>
#include <random>
#include <numeric>
#include <climits>

#include <unistd.h>
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

void GX_Utils :: addMatrix( GX_DataMatrix * dest, const GX_DataMatrix & src )
{
	assert( dest->size() == src.size() );

	for( size_t i = 0; i < src.size(); i++ ) {

		GX_DataVector & destVec = dest->at( i );
		const GX_DataVector & srcVec = src[ i ];

		assert( destVec.size() == srcVec.size() );

		for( size_t j = 0; j < srcVec.size(); j++ ) {
			destVec[ j ] += srcVec[ j ];
		}
	}
}

void GX_Utils :: printMnistImage( const char * tag, const GX_DataVector & data )
{
	printf( "%s { %ld }\n", tag, data.size() );

	for( size_t i = 0; i < 28; i++ ) {
		for( size_t j = 0; j < 28; j++ ) {
			size_t idx = i * 28 + j;
			printf( "%s ",  idx < data.size() && data[ idx ] != 0 ? "1" : "0" );
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

		images->push_back( GX_DataVector( imageSize ) );
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

		labels->push_back( GX_DataVector( maxClasses ) );
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

	fprintf( fp, "Network: LayerCount = %ld, LossFuncType = %d\n",
			network.getLayers().size(), network.getLossFuncType() );

	for( size_t i = 0; i < network.getLayers().size(); i++ ) {
		GX_Layer * layer = network.getLayers() [ i ];

		const GX_NeuronPtrVector & neurons = layer->getNeurons();

		fprintf( fp, "Layer#%ld: NeuronCount = %ld, ActFuncType = %d\n",
				i, neurons.size(), layer->getActFuncType() );

		for( size_t j = 0; j < neurons.size(); j++ ) {
			GX_Neuron * neuron = neurons[ j ];
			const GX_DataVector & weights = neuron->getWeights();

			fprintf( fp, "Neuron#%ld: WeightCount = %ld\n", j , weights.size() );
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
	auto getNumber = []( std::string const & line, const char * fmt, double defaultValue ) {
		double value = defaultValue;

		std::regex ex( fmt );
		std::smatch match;

		if( std::regex_search( line, match, ex ) ) value = std::stod( match[ match.size() - 1 ] );

		return value;
	};

	auto getNumberVector = []( std::string const & line, std::string * prefix, GX_DataVector * data ) {
		const std::regex comma( "," );

		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), comma, -1 ),
			std::sregex_token_iterator() };

		data->resize( srow.size() );
		for( size_t i = 0; i < srow.size(); i++ ) {
			( *data )[ i ] = std::stod( srow[ i ] );
		}
	};

	std::ifstream fp( path );

	if( !fp ) return false;

	std::string line, name;

	// Network: xxx
	if( ! std::getline( fp, line ) ) return false;

	network->setLossFuncType( getNumber( line, "LossFuncType = (\\d+)", GX_Network::eMeanSquaredError ) );

	int layerCount = getNumber( line, "LayerCount = (\\d+)", 0 );

	network->getLayers().reserve( layerCount );

	for( int i = 0; i < layerCount; i++ ) {
		// Layer#x: xxx
		if( ! std::getline( fp, line ) ) return false;

		int neuronCount = getNumber( line, "NeuronCount = (\\d+)", 0 );
		int actFuncType = getNumber( line, "ActFuncType = (\\d+)", GX_Layer::eSigmoid );

		GX_Layer * layer = NULL;

		for( int j = 0; j < neuronCount; j++ ) {

			// Neuron#x: xxx
			if( ! std::getline( fp, line ) ) return false;

			int weightCount = getNumber( line, "WeightCount = (\\d+)", 0 );

			if( NULL == layer )  layer = new GX_Layer( neuronCount, weightCount, actFuncType );

			GX_Neuron * neuron = layer->getNeurons()[ j ];

			// Bias#x: xxx
			if( ! std::getline( fp, line ) ) return false;

			neuron->setBias( getNumber( line, ": (-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)", 0 ) );

			// Weights#x: xxx:
			if( ! std::getline( fp, line ) ) return false;

			// xxx, xxx, xxx
			if( ! std::getline( fp, line ) ) return false;

			getNumberVector( line, NULL, &neuron->getWeights() );
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

	while( ( c = getopt ( argc, argv, "a:t:c:e:l:b:p:dv" )) != EOF ) {
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
			case 'p':
				args->mModelPath = optarg;
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
				printf( "\t-p <model path> if path exist, then continue training\n" );
				printf( "\t-d debug mode on\n" );
				printf( "\t-v show usage\n" );
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

