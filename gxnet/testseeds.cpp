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
#include <stdio.h>

/*
* Load comma separated values from file and normalize the values
*/
bool loadData( const char * filename, GX_DataMatrix * data, std::set< int > * labels )
{
	const std::regex comma(",");

	std::ifstream fp( filename );

	if( !fp ) return false;

	std::string line;

	while( fp && std::getline( fp, line ) ) {
		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), comma, -1 ),
			std::sregex_token_iterator() };

		data->push_back( GX_DataVector() );
		data->back().resize( srow.size(), 0 );

		std::transform( srow.begin(), srow.end(), data->back().begin(),
				[](std::string const& val) {return std::stof(val); } );

		labels->insert( std::stoi( srow.back() ) );
	}

	GX_DataVector min( data[ 0 ].size(), FLT_MAX ), max( data[ 0 ].size(), FLT_MIN );;

	// normalize data
	{
		for( auto & item : * data ) {
			for( size_t i = 0; i < item.size(); i++ ) {
				if( item[ i ] > max[ i ] ) max[ i ] = item[ i ];
				if( item[ i ] < min[ i ] ) min[ i ] = item[ i ];
			}
		}

		for( auto & item : * data ) {
			// skip last element, it's the label
			for( size_t i = 0; i < item.size() - 1; i++ ) {
				item[ i ] = ( item[ i ] - min[ i ] ) / ( max[ i ] - min[ i ] );
			}
		}
	}

	return true;
}

void check( const char * tag, GX_Network & network, GX_DataMatrix & input, GX_DataMatrix & target, bool isDebug )
{
	if( isDebug ) network.print();

	int correct = 0;

	for( size_t i = 0; i < input.size(); i++ ) {

		GX_DataMatrix output;

		bool ret = network.forward( input[ i ], &output );

		int outputType = GX_Utils::max_index( output.back().begin(), output.back().end() );
		int targetType = GX_Utils::max_index( target[ i ].begin(), target[ i ].end() );

		if( isDebug ) printf( "forward %d, index %zu, %d %d\n", ret, i, outputType, targetType );

		if( outputType == targetType ) correct++;

		for( size_t j = 0; isDebug && j < output.back().size(); j++ ) {
			printf( "\t%zu %.8f %.8f\n", j, output.back()[ j ], target[ i ][ j ] );
		}
	}

	printf( "GX_Network %s, %d/%ld = %.2f\n", tag, correct, input.size(), ((float)correct) / input.size() );
}

void splitData( const CmdArgs_t & args, const GX_DataMatrix & data, const std::set< int > & labels,
		GX_DataMatrix * input, GX_DataMatrix * target,
		GX_DataMatrix * input4eval, GX_DataMatrix * target4eval )
{
	std::vector< int > idxOfData( data.size() );
	std::iota( idxOfData.begin(), idxOfData.end(), 0 );

	std::map< int, int > mapOflabels;
	for( auto & item : labels ) mapOflabels[ item ] = mapOflabels.size();

	for( int i = args.mEvalCount; i > 0; i-- ) {
		int n = std::rand() % idxOfData.size();

		input4eval->push_back( data[ idxOfData[ n ] ] );
		input4eval->back().pop_back(); // remove last element, it's the label
		
		target4eval->push_back( GX_DataVector() );
		target4eval->back().resize( mapOflabels.size(), 0 );
		target4eval->back()[ mapOflabels[ data[ idxOfData[ n ] ].back() ] ] = 1;

		idxOfData.erase( idxOfData.begin() + n );
	}

	for( size_t i = 0; i < idxOfData.size(); i++ ) {
		input->push_back( data[ idxOfData[ i ] ] );
		input->back().pop_back(); // remove last element, it's the label

		target->push_back( GX_DataVector() );
		target->back().resize( mapOflabels.size(), 0 );
		target->back()[ mapOflabels[ data[ idxOfData[ i ] ].back() ] ] = 1;
	}
}

void test( const CmdArgs_t & args )
{
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	GX_DataMatrix data;
	std::set< int > labels;

	loadData( "seeds_dataset.csv", &data, &labels );

	GX_DataMatrix input, target, input4eval, target4eval;

	splitData( args, data, labels, &input, &target, &input4eval, &target4eval );

	const char * path = "./seeds.model";

	// train & check & save
	{
		GX_Network network;

		network.addLayer( 5, input[ 0 ].size() );
		network.addLayer( target[ 0 ].size(), 5 );

		check( "before train", network, input4eval, target4eval, args.mIsDebug );

		bool ret = network.train( input, target, args.mIsShuffle,
			args.mEpochCount, args.mMiniBatchCount, args.mLearningRate, args.mLambda );

		GX_Utils::save( path, network );

		printf( "train %s\n", ret ? "succ" : "fail" );

		check( "after train", network, input4eval, target4eval, args.mIsDebug );
	}

	{
		GX_Network network;

		GX_Utils::load( path, &network );

		check( "load model", network, input4eval, target4eval, args.mIsDebug );
	}
}

int main( const int argc, char * argv[] )
{
	CmdArgs_t defaultArgs = {
		.mEvalCount = 42,
		.mEpochCount = 10,
		.mMiniBatchCount = 1,
		.mLearningRate = 0.3,
		.mIsDebug = false,
		.mIsShuffle = true,
	};

	CmdArgs_t args = defaultArgs;

	GX_Utils::getCmdArgs( argc, argv, defaultArgs, &args );

	test( args );

	return 0;
}

