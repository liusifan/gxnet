#include "gxutils.h"
#include "gxnet.h"
#include "gxact.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <set>
#include <limits.h>
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

		data->push_back( GX_DataVector( srow.size() ) );

		std::transform( srow.begin(), srow.end(), std::begin( data->back() ),
				[](std::string const& val) {return std::stof(val); } );

		labels->insert( std::stoi( srow.back() ) );
	}

	GX_DataVector min, max;
	min.resize( data[ 0 ].size(), FLT_MAX );
	max.resize( data[ 0 ].size(), 1.0 * INT_MIN );

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

template< typename NetworkType >
void check( const char * tag, NetworkType & network, GX_DataMatrix & input, GX_DataMatrix & target, bool isDebug )
{
	if( isDebug ) network.print();

	int correct = 0;

	for( size_t i = 0; i < input.size(); i++ ) {

		GX_DataMatrix output;

		bool ret = network.forward( input[ i ], &output );

		int outputType = GX_Utils::max_index( std::begin( output.back() ), std::end( output.back() ) );
		int targetType = GX_Utils::max_index( std::begin( target[ i ] ), std::end( target[ i ] ) );

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

		const GX_DataVector & item = data[ idxOfData[ n ] ];

		// remove last element, it's the label
		input4eval->push_back( GX_DataVector( item.size() - 1 ) );
		std::copy( std::begin( item ), std::end( item ) - 1, std::begin( input4eval->back() ) );

		target4eval->push_back( GX_DataVector() );
		target4eval->back().resize( mapOflabels.size(), 0 );
		target4eval->back()[ mapOflabels[ item[ item.size() - 1 ] ] ] = 1;

		idxOfData.erase( idxOfData.begin() + n );
	}

	for( size_t i = 0; i < idxOfData.size(); i++ ) {

		const GX_DataVector & item = data[ idxOfData[ i ] ];

		// remove last element, it's the label
		input->push_back( GX_DataVector( item.size() - 1 ) );
		std::copy( std::begin( item ), std::end( item ) - 1, std::begin( input->back() ) );

		target->push_back( GX_DataVector() );
		target->back().resize( mapOflabels.size(), 0 );
		target->back()[ mapOflabels[ item[ item.size() - 1 ] ] ] = 1;
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

		network.setShuffle( args.mIsShuffle );
		network.setLossFuncType( GX_Network::eCrossEntropy );

		GX_BaseLayer * layer = NULL;

		layer = new GX_FullConnLayer( 5, input[ 0 ].size() );
		layer->setActFunc( GX_ActFunc::sigmoid() );
		network.addLayer( layer );

		layer = new GX_FullConnLayer( target[ 0 ].size(), layer->getOutputSize() );
		layer->setActFunc( GX_ActFunc::softmax() );
		network.addLayer( layer );

		check( "before train", network, input4eval, target4eval, args.mIsDebug );

		bool ret = network.train( input, target,
			args.mEpochCount, args.mMiniBatchCount, args.mLearningRate, args.mLambda );

		GX_Utils::save( path, network );

		printf( "train %s\n", ret ? "succ" : "fail" );

		check( "after train", network, input4eval, target4eval, args.mIsDebug );
	}

	{
		GX_Network network;

		GX_Utils::load( path, &network );

		network.print();

		check( "load model", network, input4eval, target4eval, args.mIsDebug );
	}
}

int main( const int argc, char * argv[] )
{
	CmdArgs_t defaultArgs = {
		.mEvalCount = 42,
		.mEpochCount = 10,
		.mMiniBatchCount = 1,
		.mLearningRate = 0.1,
		.mIsDebug = false,
		.mIsShuffle = true,
	};

	CmdArgs_t args = defaultArgs;

	GX_Utils::getCmdArgs( argc, argv, defaultArgs, &args );

	test( args );

	return 0;
}

