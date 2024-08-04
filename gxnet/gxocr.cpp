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
#include <getopt.h>

bool readImage( const char * path, GX_DataVector * input )
{
	auto getNumberVector = []( std::string const & line, GX_DataVector * data ) {
		const std::regex colon( "," );

		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), colon, -1 ),
			std::sregex_token_iterator() };

		data->resize( srow.size(), 0 );
		for( size_t i = 0; i < srow.size(); i++ ) {
			( *data )[ i ] = std::stod( srow[ i ] ) / 255.0;
		}
	};

	std::ifstream fp( path );

	if( !fp ) {
		printf( "open %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	std::string line;

	if( ! std::getline( fp, line ) ) {
		printf( "getline %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	getNumberVector( '[' == line[ 0 ] ? line.c_str() + 1 : line, input );

	//printf( "%s read %s, size %zu\n", __func__, path, input->size() );

	return true;
}

int test( const char * modelFile, const char * imgFile )
{
	GX_DataVector input;
	GX_DataMatrix output;

	if( ! readImage( imgFile, &input ) ) return -1;

	GX_Network network;

	if( ! GX_Utils::load( modelFile, &network ) ) return -1;

	if( input.size() < network.getLayers()[ 0 ]->getInputSize() ) {
		GX_DataVector newInput;
		GX_Utils::expandMnistImage( input, &newInput );
		input = newInput;
	}

	bool ret = network.forward( input, &output );

	if( ! ret ) {
		printf( "forward fail\n" );
		return -1;
	}

	int result = GX_Utils::max_index( std::begin( output.back() ), std::end( output.back() ) );

	printf( "%s    \t-> %d, nn.output %f\n", imgFile, result, output.back()[ result ] );

	return result;
}

void usage( const char * name )
{
	printf( "%s --model <model file> --file <mnist file>\n", name );
}

int main( const int argc, char * argv[] )
{
	static struct option opts[] = {
		{ "model",   required_argument,  NULL, 1 },
		{ "file",  required_argument,  NULL, 2 },
		{ 0, 0, 0, 0}
	};

	char * model = NULL, * file = NULL;

	int c = 0;
	while( ( c = getopt_long( argc, argv, "", opts, NULL ) ) != EOF ) {
		switch( c ) {
			case 1:
				model = optarg;
				break;
			case 2:
				file = optarg;
				break;
			default:
				usage( argv[ 0 ] );
				break;
		}
	}

	if( NULL == model || NULL == file ) {
		usage( argv[ 0 ] );
		return 0;
	}

	int ret = test( model, file );

	return ret;
}

