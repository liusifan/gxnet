
#include "gxeval.h"
#include "gxutils.h"

void gx_eval( const char * tag, GX_Network & network, GX_DataMatrix & input, GX_DataMatrix & target, bool isDebug )
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

		int outputType = GX_Utils::max_index( std::begin( output.back() ), std::end( output.back() ) );
		int targetType = GX_Utils::max_index( std::begin( target[ i ] ), std::end( target[ i ] ) );

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

