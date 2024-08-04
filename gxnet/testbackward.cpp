
#include "gxnet.h"
#include "gxutils.h"
#include "gxact.h"

#include <stdio.h>

void check( const char * tag, GX_Network & network, GX_DataVector & input, GX_DataVector & target )
{
	printf( "%s\n", tag );

	network.print( true );

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

const char * GRAD_EXAMPLES =
R"(
input { 2 }
3.00000000e+00 1.00000000e+00

gradient { 4 }
#0 -5.343815e-08 -1.781272e-08
#1 1.420144e-03 4.733812e-04
#2 -1.051877e-01 -1.891928e-03
#3 2.654905e-02 4.775168e-04

input { 2 }
-1.00000000e+00 4.00000000e+00

gradient { 4 }
#0 -2.717464e-07 1.086986e-06
#1 -1.962391e-12 7.849563e-12
#2 2.301185e-07 2.767417e-01
#3 -2.081411e-08 -2.503115e-02

batch gradient { 4 }
#0 -3.251846e-07 1.069173e-06
#1 1.420144e-03 4.733812e-04
#2 -1.051875e-01 2.748497e-01
#3 2.654902e-02 -2.455364e-02

('g_W', array([[-3.25184585e-07,  1.42014367e-03],
       [ 1.06917303e-06,  4.73381232e-04]]))
('g_V', array([[-0.10518746,  0.02654902],
       [ 0.27484974, -0.02455364]]))

)";

const char * TRAIN_EXAMPLES =
R"(
{{{ isDetail true
Network: LayerCount = 2; LossFuncType = 1;

Layer#0: Type = 4; ActFuncType = 1; InputDims = 2; OutputDims = 2;
Weights: Count = 2; InputCount = 2;
        Neuron#0: WeightCount = 2, Bias = 0.00000000
                Weight#0: 6.00000016
                Weight#1: -2.00000053
        Neuron#1: WeightCount = 2, Bias = 0.00000000
                Weight#0: -3.00071007
                Weight#1: 4.99976331
Layer#1: Type = 4; ActFuncType = 1; InputDims = 2; OutputDims = 2;
Weights: Count = 2; InputCount = 2;
        Neuron#0: WeightCount = 2, Bias = 0.00000000
                Weight#0: 1.05259373
                Weight#1: 0.11257513
        Neuron#1: WeightCount = 2, Bias = 0.00000000
                Weight#0: -2.01327451
                Weight#1: 2.01227682
}}}
forward 1, input { 2 }, output { 2 }
layer 0
        0.99999989      0.01794445
layer 1
        0.74165987      0.12162137
sse 0.08153138

('net.V', array([[ 1.05259373, -2.01327451],
       [ 0.11257513,  2.01227682]]))
('net.W', array([[ 6.00000016, -3.00071007],
       [-2.00000053,  4.99976331]]))
('y', array([[0.74165987, 0.12162137]]))
)";

void testNetwork()
{
	//https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html

	GX_DataMatrix v = { { 6, -2 }, { -3, 5 } };
	GX_DataMatrix w = { { 1, 0.25 }, { -2, 2 } };
	GX_DataVector b = { 0, 0 };

	GX_DataMatrix input = { { 3, 1}, { -1, 4 } };
	GX_DataMatrix target = { { 1, 0 }, { 0, 1 } };

	GX_DataType learningRate = 0.5;

	GX_Network network( GX_Network::eMeanSquaredError );
	{
		network.setDebug( true );
		network.setShuffle( false );

		GX_FullConnLayer * layer = NULL;

		layer = new GX_FullConnLayer( 2, 2 );
		layer->setWeights( v, b );
		layer->setActFunc( GX_ActFunc::sigmoid() );
		network.addLayer( layer );

		layer = new GX_FullConnLayer( 2, 2 );
		layer->setWeights( w, b );
		layer->setActFunc( GX_ActFunc::sigmoid() );
		network.addLayer( layer );
	}

	check( "before train", network, input[ 0 ], target[ 0 ] );

	printf( "\e[0;31m%s\e[0m\n", GRAD_EXAMPLES );

	network.train( input, target, 1, 2, learningRate, 0, NULL );

	check( "after train", network, input[ 0 ], target[ 0 ] );

	printf( "\e[0;31m%s\e[0m\n", TRAIN_EXAMPLES );
}

int main()
{
	testNetwork();

	return 0;
}

