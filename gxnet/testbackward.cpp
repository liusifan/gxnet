
#include "gxnet.h"
#include "gxutils.h"

void check( const char * tag, GX_Network & network, GX_DataVector & input, GX_DataVector & target )
{
	printf( "%s\n", tag );

	network.print();

	GX_DataMatrix output;

	bool ret = network.forward( input, &output );

	printf( "forward %d, input { %ld }, output { %ld }\n", ret, input.size(), output.size() );
	for( auto & i : output ) {
		printf( "layer %ld\n", &i - &( output[ 0 ] ) );
		for( auto & j : i ) {
			printf( "\t%.8f ", j );
		}
		printf( "\n" );
	}

	GX_DataType sse = GX_Utils::calcSSE( output.back(), target );
	printf( "sse %.8f\n", sse );
}

void testBackward()
{
	//https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html

	GX_DataMatrix v = { { 6, -2 }, { -3, 5 } };
	GX_DataMatrix w = { { 1, 0.25 }, { -2, 2 } };
	GX_DataVector b = { 0, 0 };

	GX_DataMatrix input = { { 3, 1}, { -1, 4 } };
	GX_DataMatrix target = { { 1, 0 }, { 0, 1 } };

	GX_DataType learningRate = 0.5;

	GX_Network network( true, true, true );

	network.addLayer( v, b );
	network.addLayer( w, b );

	check( "before train", network, input[ 0 ], target[ 0 ] );
	network.train( input, target, true, 1, 2, learningRate );
	check( "after train", network, input[ 0 ], target[ 0 ] );
}

int main()
{
	testBackward();

	return 0;
}

