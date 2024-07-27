
#include "gxlayer.h"

#include "gxutils.h"

#include <cstdio>

GX_DataVector input = {
	0,  50, 0,  29,
	0,  80, 31, 2,
	33, 90, 0,  75,
	0,  9,  0,  95
};

void testConvLayer()
{
	GX_DataVector filter1 = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	GX_DataVector filter2 = {
		1,  2,  1,
		0,  0,  0,
		-1, -2, 1
	};


	GX_DataVector filters( filter1.size() + filter2.size() );
	std::copy( std::begin( filter1 ), std::end( filter1 ), std::begin( filters ) );
	std::copy( std::begin( filter2 ), std::end( filter2 ), std::begin( filters ) + filter1.size() );

	GX_DataVector input2( 2 * input.size() );
	std::copy( std::begin( input ), std::end( input ), std::begin( input2 ) );
	std::copy( std::begin( input ), std::end( input ), std::begin( input2 ) + input.size() );

	GX_DataVector biases = { 1 };

	GX_Dims filterDims = { 1, 2, 3, 3 };
	GX_Dims inputDims = { 2, 4, 4 };

	GX_Utils::printVector( "input", input2, inputDims, false );

	GX_Utils::printVector( "filters", filters, false );

	GX_DataVector output;

	GX_ConvLayer conv( inputDims, filters, filterDims, biases );
	conv.setDebug( true );

	conv.print( true );

	conv.forward( input2, &output );

	GX_Utils::printVector( "conv.output", output, conv.getOutputDims(), false );
	GX_Utils::printVector( "conv.output", output, false );

	GX_DataVector outDelta( output.size() ), inDelta( input2.size() );

	for( size_t i = 0; i < outDelta.size(); i++ ) outDelta[ i ] = ( i + 1 ) * 0.1;

	conv.backward( input2, output, &outDelta, &inDelta );

	GX_Utils::printVector( "conv.outDelta", outDelta, conv.getOutputDims(), false );

	GX_Utils::printVector( "conv.inDelta", inDelta, inputDims, false );

	GX_DataMatrix gradient;
	conv.initGradientMatrix( &gradient );

	GX_DataMatrix::iterator iter = gradient.begin();

	conv.collectGradient( input2, output, outDelta, &iter );

	GX_Utils::printVector( "filters.grad", gradient[ 0 ], filterDims, false );

	GX_DataMatrix::const_iterator constIter = gradient.begin();
	conv.applyGradient( outDelta, &constIter, 1, 0.1, 1, 1 );

	conv.print( true );
}

void testMaxPoolLayer()
{
	GX_Dims inputDims = { 1, 4, 4 };
	GX_Utils::printVector( "input", input, inputDims, false );

	GX_DataVector output;

	GX_MaxPoolLayer maxpool( inputDims, 2 );
	maxpool.forward( input, &output );

	GX_Utils::printVector( "maxpool.output", output, maxpool.getOutputDims(), false );

	GX_DataVector outDelta( output.size() ), inDelta( input.size() );

	outDelta = 0.5;

	maxpool.backward( input, output, &outDelta, &inDelta );

	GX_Utils::printVector( "maxpool.inDelta", inDelta, maxpool.getInputDims(), false );
}

int main( int argc, const char * argv[] )
{
	testConvLayer();

	//testMaxPoolLayer();

	return 0;
}

