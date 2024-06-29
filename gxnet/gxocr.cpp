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

	GX_DataVector * realInput = &input;

	bool ret = network.forward( *realInput, &output );

	if( ! ret ) {
		printf( "forward fail\n" );
		return -1;
	}

	int result = GX_Utils::max_index( output.back().begin(), output.back().end() );

	printf( "%s    \t-> %d, nn.output %f\n", imgFile, result, output.back()[ result ] );

	if( output.back()[ result ] < 0.5 ) {
		//GX_DataType outputTotal = std::accumulate( output.back().begin(), output.back().end(), 0.0 );
		for( size_t i = 0; i < output.back().size(); i++ ) {
			//printf( "\t%zu %.2f %.2f\n", i, output.back()[ i ], output.back()[ i ] / outputTotal );
		}
	}

	return result;
}

int main( const int argc, char * argv[] )
{
	if( argc < 3 ) {
		printf( "%s <mnist model file> <bmp file>\n", argv[ 0 ] );
		return -1;
	}

	openlog( argv[0], LOG_PID | LOG_CONS | LOG_PERROR | LOG_NDELAY, LOG_USER );

	setlogmask(LOG_UPTO(LOG_DEBUG));

	int ret = test( argv[1], argv[2] );

	closelog();

	return ret;
}

