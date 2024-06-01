
#include "gxnet.h"
#include "gxutils.h"

#include <syslog.h>
#include <time.h>
#include <assert.h>

#include <random>
#include <numeric>

GX_Neuron :: GX_Neuron( const int weightCount )
{
	for( int i = 0; i < weightCount; i++ ) {
		mWeights.push_back( GX_Utils::random() );
	}
	mBias = GX_Utils::random();
}

GX_Neuron :: ~GX_Neuron()
{
}

GX_DataVector & GX_Neuron :: getWeights()
{
	return mWeights;
}

const GX_DataType GX_Neuron :: getBias()
{
	return mBias;
}

void GX_Neuron :: setBias( GX_DataType bias )
{
	mBias = bias;
}

bool GX_Neuron :: calcOutput( const GX_DataVector input, bool isDebug,
		bool ignoreBias, GX_DataType * netOutput, GX_DataType * output )
{
	if( input.size() < mWeights.size() ) {
		syslog( LOG_ERR, "%s input.size %zu mWeights.size %zu",
				__func__, input.size(), mWeights.size() );
		printf( "%s input.size %zu mWeights.size %zu\n",
				__func__, input.size(), mWeights.size() );
		return false;
	}

	*netOutput = ignoreBias ? 0 : mBias;

	for( size_t i = 0; i < mWeights.size(); i++ ) {
		GX_DataType tmp = mWeights[ i ] * input[ i ];

		//if( isDebug ) printf( "output %e = %e * %e\n", tmp, mWeights[ i ], input[ i ] );

		*netOutput += tmp;
	}

	//if( isDebug ) printf( "netOutput %e\n", *netOutput );

	*output = 1.0f / ( 1.0f + std::exp( - ( *netOutput ) ) );

	//if( isDebug ) printf( "output %e = 1.0 / ( 1.0 + std::exp( -1 * %e ) \n", *output, *netOutput );

	return true;
}

////////////////////////////////////////////////////////////

GX_Layer :: GX_Layer( const int neuronCount, const int weightCount )
{
	for( int i = 0; i < neuronCount; i++ ) {
		mNeurons.push_back( new GX_Neuron( weightCount ) );
	}
}

GX_Layer :: ~GX_Layer()
{
	for( auto & neuron : mNeurons ) delete neuron;
}

GX_NeuronPtrVector & GX_Layer :: getNeurons()
{
	return mNeurons;
}

////////////////////////////////////////////////////////////

GX_Network :: GX_Network( bool isDebug, bool ignoreBias, bool isSumMiniBatchGrad )
{
	mIsDebug = isDebug;
	mIgnoreBias = ignoreBias;
	mIsSumMiniBatchGrad = isSumMiniBatchGrad;
}

GX_Network :: ~GX_Network()
{
	for( auto & layer : mLayers ) delete layer;
}

bool GX_Network :: addLayer( const int neuronCount, const int weightCount )
{
	if( neuronCount <= 0 || weightCount <= 0 ) return false;

	mLayers.push_back( new GX_Layer( neuronCount, weightCount ) );

	return true;
}

bool GX_Network :: addLayer( const GX_DataMatrix & weights, const GX_DataVector & bias )
{
	GX_Layer * layer = new GX_Layer( weights.size(), weights[ 0 ].size() );

	for( size_t i = 0; i < weights.size(); i++ ) {
		layer->getNeurons()[ i ]->getWeights() = weights[ i ];
		layer->getNeurons()[ i ]->setBias( bias[ i ] );
	}

	mLayers.push_back( layer );

	return true;
}

const GX_LayerPtrVector & GX_Network ::getLayers() const
{
	return mLayers;
}

GX_LayerPtrVector & GX_Network ::getLayers()
{
	return mLayers;
}

void GX_Network ::print()
{
	printf( "LayerCount: %zu\n", mLayers.size() );
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_Layer * layer = mLayers[ i ];

		printf( "Layer %ld: { %ld }\n", i, layer->getNeurons().size() );

		for( size_t j = 0; j < layer->getNeurons().size(); j++ ) {
			GX_Neuron * neuron = layer->getNeurons()[ j ];

			printf( "\tNeuron %ld: { %ld }\n", j, neuron->getWeights().size() );

			if( j > 10 ) {
				printf( "\t\t......\n" );
				break;
			}

			printf( "\t\tbias %.8f\n", neuron->getBias() );

			for( size_t k = 0; k < neuron->getWeights().size(); k++ ) {
				printf( "\t\tWeight %ld: %.8f\n", k, neuron->getWeights()[ k ] );

				if( k > 10 ) {
					printf( "\t\t......\n" );
					break;
				}
			}
		}
	}
}

bool GX_Network :: forward( const GX_DataVector & input, GX_DataMatrix * output )
{
	if( mLayers.size() <= 0 || mLayers[ 0 ]->getNeurons().size() <= 0 ) {
		printf( "%s layers.size %zu, layer[0].size %zu\n",
			__func__, mLayers.size(),  mLayers[ 0 ]->getNeurons().size() );
		return false;
	}

	if( input.size() < mLayers[ 0 ]->getNeurons()[ 0 ]->getWeights().size() ) {
		syslog( LOG_ERR, "%s input.size %zu, weights.size %zu",
				__func__, input.size(), mLayers[ 0 ]->getNeurons()[ 0 ]->getWeights().size() );
		printf( "%s input.size %zu, weights.size %zu\n",
				__func__, input.size(), mLayers[ 0 ]->getNeurons()[ 0 ]->getWeights().size() );
		return false;
	}

	const GX_DataVector * currInput = &input;

	output->reserve( mLayers.size() );

	for( auto & layer : mLayers ) {
		output->push_back( GX_DataVector() );
		output->back().reserve( layer->getNeurons().size() );

		if( output->size() > 1 ) currInput = &( (*output)[ output->size() - 2 ] );
		for( auto & neuron : layer->getNeurons() ) {
			GX_DataType netOutput = 0, tmpOutput = 0;
			if( ! neuron->calcOutput( *currInput, mIsDebug, mIgnoreBias, &netOutput, &tmpOutput ) ) return false;

			output->back().push_back( tmpOutput );
		}
	}

	return true;
}

bool GX_Network :: backward( const GX_DataVector & target,
		const GX_DataMatrix & output, GX_DataMatrix * delta )
{
	const GX_DataVector & last = output.back();

	delta->reserve( output.size() );
	for( const auto & item : output ) {
		delta->push_back( GX_DataVector() );
		delta->reserve( item.size() );
	}

	GX_Layer * layer = mLayers.back();
	for( size_t i = 0; i < layer->getNeurons().size(); i++ ) {
		GX_DataType tmpError = 2 * ( last[ i ] - target[ i ] );
		GX_DataType derivative = last[ i ] * ( 1 - last[ i ] );

		delta->back().push_back( tmpError * derivative );
	}

	for( int currLayer = mLayers.size() - 2; currLayer >= 0 ; currLayer-- ) {
		layer = mLayers[ currLayer ];

		for( size_t currNeuron = 0; currNeuron < layer->getNeurons().size(); currNeuron++ ) {
			GX_DataType derivative = output[ currLayer ][ currNeuron ] * ( 1 - output[ currLayer ][ currNeuron ] );

			GX_DataType tmpError = 0;

			GX_NeuronPtrVector & nextLayerNeurons = mLayers[ currLayer + 1 ]->getNeurons();
			for( size_t k = 0; k < nextLayerNeurons.size(); k++ ) {
				tmpError += nextLayerNeurons[ k ]->getWeights()[ currNeuron ] * ( *delta )[ currLayer + 1 ][ k ];
			}
			( *delta )[ currLayer ].push_back( tmpError * derivative );
		}
	}

	return true;
}

bool GX_Network :: collect( const GX_DataVector & input, const GX_DataMatrix & output,
			const GX_DataMatrix & delta, GX_DataMatrix * grad ) const
{
	const GX_DataVector * currInput = &input;

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		if( i > 0 ) currInput = &( output[ i - 1 ] );

		const GX_DataVector & currDelta = delta[ i ];

		GX_Layer * layer = mLayers[ i ];

		for( size_t j = 0; j < layer->getNeurons().size(); j++ ) {
			GX_Neuron * neuron = layer->getNeurons()[ j ];
			GX_DataVector & weights = neuron->getWeights();

			grad->push_back( GX_DataVector() );
			grad->reserve( layer->getNeurons().size() );

			for( size_t k = 0; k < weights.size(); k++ ) {
				GX_DataType tmp = currDelta[ j ] * ( *currInput )[ k ];
				grad->back().push_back( tmp );
			}
		}
	}

	if( mIsDebug ) {
		GX_Utils::printVector( "input", input );
		GX_Utils::printMatrix( "output", output );
		GX_Utils::printMatrix( "delta", delta );
		GX_Utils::printMatrix( "grad", *grad );
	}

	return true;
}

bool GX_Network :: apply( const GX_DataMatrix & delta, const GX_DataMatrix & grad,
		int miniBatchCount, GX_DataType learningRate, GX_DataType lambda, int trainingCount )
{
	int gradIdx = 0;

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_Layer * layer = mLayers[ i ];

		const GX_DataVector & currDelta = delta[ i ];

		for( size_t j = 0; j < layer->getNeurons().size(); j++ ) {
			GX_Neuron * neuron = layer->getNeurons()[ j ];
			GX_DataVector & weights = neuron->getWeights();

			for( size_t k = 0; k < weights.size(); k++ ) {
				weights[ k ] = ( 1 - learningRate * lambda / trainingCount ) * weights[ k ] 
						- ( grad[ gradIdx ] )[ k ] * learningRate / miniBatchCount;
			}

			neuron->setBias( neuron->getBias() - learningRate * currDelta[ j ] / miniBatchCount );

			gradIdx++;
		}
	}

	return true;
}

bool GX_Network :: train( const GX_DataMatrix & input, const GX_DataMatrix & target,
		int epochCount, GX_DataType learningRate, GX_DataType lambda )
{
	return train( input, target, false, epochCount, 1, learningRate, lambda );
}

bool GX_Network :: train( const GX_DataMatrix & input, const GX_DataMatrix & target,
		bool isShuffle, int epochCount, int miniBatchCount, GX_DataType learningRate,
		GX_DataType lambda )
{
	if( input.size() != target.size() ) return false;

	int logInterval = epochCount / 10;

	for( int n = 0; n < epochCount; n++ ) {

		std::vector< int > idxOfData( input.size() );
		std::iota( idxOfData.begin(), idxOfData.end(), 0 );
		if( isShuffle ) std::random_shuffle( idxOfData.begin(), idxOfData.end() );

		GX_DataType totalError = 0;

		miniBatchCount = std::max( miniBatchCount, 1 );

		for( size_t begin = 0; begin < idxOfData.size(); ) {
			size_t end = std::min( idxOfData.size(), begin + miniBatchCount );

			GX_DataMatrix miniBatchDelta, miniBatchGrad;

			for( size_t i = begin; i < end; i++ ) {
				GX_DataMatrix output, delta, grad;

				const GX_DataVector & currInput = input[ idxOfData[ i ] ];
				const GX_DataVector & currTarget = target[ idxOfData[ i ] ];

				if( ! forward( currInput, &output ) ) return false;

				if( ! backward( currTarget, output, &delta ) ) return false;

				if( ! collect( currInput, output, delta, &grad ) ) return false;

				GX_Utils::addMatrix( &miniBatchDelta, delta );
				GX_Utils::addMatrix( &miniBatchGrad, grad );

				GX_DataType sse = GX_Utils::calcSSE( output.back(), currTarget );

				totalError += sse;

				if( mIsDebug )  printf( "DEBUG: input #%ld sse %.8f totalError %.8f\n", i, sse, totalError );
			}

			if( ! apply( miniBatchDelta, miniBatchGrad, mIsSumMiniBatchGrad ? 1 : ( end - begin ),
					learningRate, lambda, input.size() ) ) return false;

			begin += miniBatchCount;
			end = begin + miniBatchCount;
		}

		if( logInterval <= 1 || ( logInterval > 1 && 0 == n % logInterval ) || n == ( epochCount - 1 ) )
			printf( "%ld [>] epoch %d, lr %f, error %.8f\n", time( NULL ), n, learningRate, totalError );

		if( mIsDebug ) print();
	}

	return true;
}

