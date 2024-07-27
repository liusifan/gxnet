
#include "gxnet.h"
#include "gxutils.h"

#include <syslog.h>
#include <time.h>
#include <assert.h>

#include <random>
#include <numeric>
#include <algorithm>
#include <vector>
#include <iostream>

GX_Neuron :: GX_Neuron( const int weightCount )
	: mWeights( weightCount )
{
	for( int i = 0; i < weightCount; i++ ) {
		mWeights[ i ] = GX_Utils::random();
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

GX_DataType GX_Neuron :: calcOutput( const GX_DataVector & input, bool isDebug, bool ignoreBias )
{
	if( input.size() < mWeights.size() ) {
		syslog( LOG_ERR, "%s input.size %zu mWeights.size %zu",
				__func__, input.size(), mWeights.size() );
		printf( "%s input.size %zu mWeights.size %zu\n",
				__func__, input.size(), mWeights.size() );
	}

	assert( input.size() == mWeights.size() );

	GX_DataType ret = ignoreBias ? 0 : mBias;

#ifdef GX_USE_VALARRAY
	ret += ( input * mWeights ).sum();
#else
	ret += std::inner_product( input.begin(), input.end(), mWeights.begin(), 0.0 );
#endif

	return ret;
}

////////////////////////////////////////////////////////////

GX_Layer :: GX_Layer( const int neuronCount, const int weightCount, int actFuncType )
{
	for( int i = 0; i < neuronCount; i++ ) {
		mNeurons.push_back( new GX_Neuron( weightCount ) );
	}

	mActFuncType = actFuncType;
}

GX_Layer :: ~GX_Layer()
{
	for( auto & neuron : mNeurons ) delete neuron;
}

GX_NeuronPtrVector & GX_Layer :: getNeurons()
{
	return mNeurons;
}

int GX_Layer :: getActFuncType() const
{
	return mActFuncType;
}

void GX_Layer :: activate( GX_DataVector * output ) const
{
	if( GX_Layer::eSigmoid == mActFuncType ) {
		for( auto & item : *output ) item = 1.0f / ( 1.0f + std::exp( -item ) );
	}

	if( GX_Layer::eReLU == mActFuncType ) {
		for( auto & item : *output ) {
			if( item < 0 ) {
				item = 0.01 * item;
			} else if( item > 1 ) {
				item = 1 + 0.01 * ( item - 1 );
			}
		}
	}

	if( GX_Layer::eTanh == mActFuncType ) {
		for( auto & item : *output ) item = std::tanh( item );
	}

	if( GX_Layer::eSoftmax == mActFuncType ) {
		GX_DataType maxValue = *std::max_element( std::begin( *output ), std::end( *output ) );

		GX_DataType total = 0;
		for( auto & item : *output ) {
			item = std::exp( item - maxValue );
			total += item;
		}

		for( auto & item : *output ) item = item / total;
	}
}

void GX_Layer :: derivative( const GX_DataVector & output, const GX_DataVector & dOutput,
		GX_DataVector * delta ) const
{
	assert( output.size() == delta->size() );

	if( GX_Layer::eSigmoid == mActFuncType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			( *delta )[ i ] = output[ i ] * ( 1 - output[ i ] );
			( *delta )[ i ] *= dOutput[ i ];
		}
	}

	if( GX_Layer::eReLU == mActFuncType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			( *delta )[ i ] = output[ i ] < 0 || output[ i ] > 1 ? 0.01 : 1;
			( *delta )[ i ] *= dOutput[ i ];
		}
	}

	if( GX_Layer::eTanh == mActFuncType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			( *delta )[ i ] = 1 - output[ i ] * output[ i ];
			( *delta )[ i ] *= dOutput[ i ];
		}
	}

	if( GX_Layer::eSoftmax == mActFuncType ) {
		for( size_t j = 0; j < output.size(); j++ ) {
			GX_DataVector dSoftmax( output.size() );
			for( size_t k = 0; k < output.size(); k++ ) {
				dSoftmax[ k ] = ( k == j ) ? output[ j ] * ( 1.0 - output[ j ] ) : -output[ k ] * output[ j ];
			}
#ifdef GX_USE_VALARRAY
			( *delta )[ j ] = ( dOutput * dSoftmax ).sum();
#else
			( *delta )[ j ] = std::inner_product( dOutput.begin(), dOutput.end(), dSoftmax.begin(), 0.0 );
#endif

		}
	}
}

void GX_Layer :: calcOutput( const GX_DataVector & input, bool isDebug,
		bool ignoreBias, GX_DataVector * output ) const
{
	assert( output->size() == mNeurons.size() );

	for( size_t j = 0; j < mNeurons.size(); j++ ) {
		( *output )[ j ] =  mNeurons[ j ]->calcOutput( input, isDebug, ignoreBias );
	}

	activate( output );
}

////////////////////////////////////////////////////////////

GX_Network :: GX_Network( int lossFuncType )
{
	mLossFuncType = lossFuncType;

	mIsDebug = false;
	mIsDebugBackward = false;
}

GX_Network :: ~GX_Network()
{
	for( auto & layer : mLayers ) delete layer;
}

void GX_Network :: setDebug( bool flag )
{
	mIsDebug = flag;
}

void GX_Network :: setDebugBackward( bool flag )
{
	mIsDebugBackward = flag;
}

bool GX_Network :: addLayer( const int neuronCount, const int weightCount, int actFuncType )
{
	if( neuronCount <= 0 || weightCount <= 0 ) return false;

	mLayers.push_back( new GX_Layer( neuronCount, weightCount, actFuncType ) );

	return true;
}

bool GX_Network :: addLayer( const GX_DataMatrix & weights, const GX_DataVector & bias, int actFuncType )
{
	GX_Layer * layer = new GX_Layer( weights.size(), weights[ 0 ].size(), actFuncType );

	for( size_t i = 0; i < weights.size(); i++ ) {
		layer->getNeurons()[ i ]->getWeights() = weights[ i ];
		layer->getNeurons()[ i ]->setBias( bias[ i ] );
	}

	mLayers.push_back( layer );

	return true;
}

void GX_Network :: setLossFuncType( int lossFuncType )
{
	mLossFuncType = lossFuncType;
}

int GX_Network :: getLossFuncType() const
{
	return mLossFuncType;
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
	printf( "Network: { %ld }, LossFuncType %d\n", mLayers.size(), mLossFuncType );
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_Layer * layer = mLayers[ i ];

		printf( "Layer#%ld: { %ld }, ActFuncType %d\n",
				i, layer->getNeurons().size(), layer->getActFuncType() );

		for( size_t j = 0; j < layer->getNeurons().size(); j++ ) {
			GX_Neuron * neuron = layer->getNeurons()[ j ];

			printf( "\tNeuron#%ld: { %ld }\n", j, neuron->getWeights().size() );

			if( j > 10 ) {
				printf( "\t\t......\n" );
				break;
			}

			printf( "\t\tbias %.8f\n", neuron->getBias() );

			for( size_t k = 0; k < neuron->getWeights().size(); k++ ) {
				printf( "\t\tWeight#%ld: %.8f\n", k, neuron->getWeights()[ k ] );

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
	assert( output->size() == 0 );

	output->reserve( mLayers.size() );

	for( auto & layer: mLayers ) {
		output->push_back( GX_DataVector( layer->getNeurons().size() ) );
	}

	return forwardInternal( input, output );
}

bool GX_Network :: forwardInternal( const GX_DataVector & input, GX_DataMatrix * output )
{
	if( mLayers.size() <= 0 || mLayers[ 0 ]->getNeurons().size() <= 0 ) {
		printf( "%s layers.size %zu, layer[0].size %zu\n",
			__func__, mLayers.size(),  mLayers[ 0 ]->getNeurons().size() );
		return false;
	}

	if( input.size() != mLayers[ 0 ]->getNeurons()[ 0 ]->getWeights().size() ) {
		syslog( LOG_ERR, "%s input.size %zu, weights.size %zu",
				__func__, input.size(), mLayers[ 0 ]->getNeurons()[ 0 ]->getWeights().size() );
		printf( "%s input.size %zu, weights.size %zu\n",
				__func__, input.size(), mLayers[ 0 ]->getNeurons()[ 0 ]->getWeights().size() );
		return false;
	}

	const GX_DataVector * currInput = &input;

	assert( output->size() == mLayers.size() );

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_Layer * layer = mLayers[ i ];

		if( i > 0 ) currInput = &( (*output)[ i - 1 ] );

		layer->calcOutput( *currInput, mIsDebug, mIsDebugBackward, &( ( *output )[ i ] ) );
	}

	return true;
}

GX_DataType GX_Network :: calcLoss( const GX_DataVector & target, const GX_DataVector & output )
{
	GX_DataType	ret = 0;

	if( eMeanSquaredError == mLossFuncType ) {
		ret = GX_Utils::calcSSE( output, target );
	}

	if( eCrossEntropy == mLossFuncType ) {
		for( size_t x = 0; x < target.size(); x++ ) {
			GX_DataType y = target[ x ], a = output[ x ];
			GX_DataType tmp = y * std::log( a ); // + ( 1 - y ) * std::log( 1 - a );
			ret -= tmp;
		}
	}

	return ret;
}

void GX_Network :: calcOutputDelta( const GX_Layer & layer, const GX_DataVector & target,
		const GX_DataVector & output, GX_DataVector * delta )
{
	assert( target.size() == output.size() );

	if( eMeanSquaredError == mLossFuncType ) {

		GX_DataVector dOutput( target.size() );
		for( size_t i = 0; i < target.size(); i++ ) {
			dOutput[ i ] = 2 * ( output[ i ] - target[ i ] );
		}

		layer.derivative( output, dOutput, delta );
	}

	if( eCrossEntropy == mLossFuncType ) {
		for( size_t i = 0; i < target.size(); i++ ) {
			( *delta )[ i ] = output[ i ] - target[ i ];
		}
	}
}

bool GX_Network :: backward( const GX_DataVector & target,
		const GX_DataMatrix & output, GX_DataMatrix * delta )
{
	const GX_DataVector & lastOutput = output.back();

	assert( delta->size() == mLayers.size() );

	GX_Layer * layer = mLayers.back();

	calcOutputDelta( *layer, target, lastOutput, &( delta->back() ) );

	for( int currLayer = mLayers.size() - 2; currLayer >= 0 ; currLayer-- ) {
		layer = mLayers[ currLayer ];

		GX_DataVector dOutput( layer->getNeurons().size() );
		for( size_t currNeuron = 0; currNeuron < layer->getNeurons().size(); currNeuron++ ) {
			GX_DataType nextLayerDelta = 0;
			GX_NeuronPtrVector & nextLayerNeurons = mLayers[ currLayer + 1 ]->getNeurons();
			for( size_t k = 0; k < nextLayerNeurons.size(); k++ ) {
				nextLayerDelta += nextLayerNeurons[ k ]->getWeights()[ currNeuron ] * ( *delta )[ currLayer + 1 ][ k ];
			}
			dOutput[ currNeuron ] = nextLayerDelta;
		}

		layer->derivative( output[ currLayer ], dOutput, &( ( *delta )[ currLayer ] ) );
	}

	return true;
}

bool GX_Network :: collect( const GX_DataVector & input, const GX_DataMatrix & output,
			const GX_DataMatrix & delta, GX_DataMatrix * grad ) const
{
	const GX_DataVector * currInput = &input;

	int gradIdx = 0;

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		if( i > 0 ) currInput = &( output[ i - 1 ] );

		const GX_DataVector & currDelta = delta[ i ];

		GX_Layer * layer = mLayers[ i ];

		for( size_t j = 0; j < layer->getNeurons().size(); j++ ) {
			GX_Neuron * neuron = layer->getNeurons()[ j ];
			GX_DataVector & weights = neuron->getWeights();

			assert( grad->size() > gradIdx );

			GX_DataVector & currGrad = ( *grad )[ gradIdx++ ];

			assert( currGrad.size() == weights.size() );

			for( size_t k = 0; k < weights.size(); k++ ) {
				currGrad[ k ] = currDelta[ j ] * ( *currInput )[ k ];
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
				if( mIsDebugBackward ) {
					weights[ k ] = weights[ k ] - grad[ gradIdx ][ k ] * learningRate;
				} else {
					weights[ k ] = ( 1 - learningRate * lambda / trainingCount ) * weights[ k ] 
							- ( grad[ gradIdx ] )[ k ] * learningRate / miniBatchCount;
				}
			}

			if( ! mIsDebugBackward )
				neuron->setBias( neuron->getBias() - learningRate * currDelta[ j ] / miniBatchCount );

			gradIdx++;
		}
	}

	return true;
}

void GX_Network :: initGradMatrix( const GX_LayerPtrVector & layers,
		GX_DataMatrix * miniBatchGrad, GX_DataMatrix * grad )
{
	assert( miniBatchGrad->size() == 0 );
	assert( grad->size() == 0 );

	int gradSize = 0;
	for( auto & layer: layers ) gradSize += layer->getNeurons().size();

	grad->reserve( gradSize );
	miniBatchGrad->reserve( gradSize );

	for( auto & layer: layers ) {
		for( auto & neuron : layer->getNeurons() ) {
			grad->push_back( GX_DataVector( neuron->getWeights().size() ) );
			miniBatchGrad->push_back( GX_DataVector( neuron->getWeights().size() ) );
		}
	}
}

void GX_Network :: initOutputAndDeltaMatrix( const GX_LayerPtrVector & layers,
		GX_DataMatrix * output, GX_DataMatrix * miniBatchDelta, GX_DataMatrix * delta )
{
	assert( output->size() == 0 );
	assert( miniBatchDelta->size() == 0 );
	assert( delta->size() == 0 );

	output->reserve( layers.size() );
	delta->reserve( layers.size() );
	miniBatchDelta->reserve( layers.size() );

	for( auto & layer: layers ) {
		output->push_back( GX_DataVector( layer->getNeurons().size() ) );
		delta->push_back( GX_DataVector( layer->getNeurons().size() ) );
		miniBatchDelta->push_back( GX_DataVector( layer->getNeurons().size() ) );
	}
}

bool GX_Network :: train( const GX_DataMatrix & input, const GX_DataMatrix & target,
		int epochCount, GX_DataType learningRate, GX_DataType lambda, GX_DataVector * losses )
{
	return train( input, target, false, epochCount, 1, learningRate, lambda, losses );
}

bool GX_Network :: train( const GX_DataMatrix & input, const GX_DataMatrix & target,
		bool isShuffle, int epochCount, int miniBatchCount, GX_DataType learningRate,
		GX_DataType lambda, GX_DataVector * losses )
{
	if( input.size() != target.size() ) return false;

	time_t beginTime = time( NULL );

	printf( "%s\tstart train, input { %zu }, target { %zu }\n",
			ctime( &beginTime ), input.size(), target.size() );

	int logInterval = epochCount / 10;

	std::random_device rd;
	std::mt19937 gen( rd() );

	GX_DataMatrix miniBatchGrad, grad;
	initGradMatrix( mLayers, &miniBatchGrad, &grad );

	GX_DataMatrix output, miniBatchDelta, delta;
	initOutputAndDeltaMatrix( mLayers, &output, &miniBatchDelta, &delta );

	if( NULL != losses ) losses->resize( epochCount, 0 );

	for( int n = 0; n < epochCount; n++ ) {

		std::vector< int > idxOfData( input.size() );
		std::iota( idxOfData.begin(), idxOfData.end(), 0 );
		if( isShuffle ) std::shuffle( idxOfData.begin(), idxOfData.end(), gen );

		GX_DataType totalLoss = 0;

		miniBatchCount = std::max( miniBatchCount, 1 );

		for( size_t begin = 0; begin < idxOfData.size(); ) {
			size_t end = std::min( idxOfData.size(), begin + miniBatchCount );

			for( auto & vec : miniBatchGrad ) std::fill( std::begin( vec ), std::end( vec ), 0.0 );
			for( auto & vec : miniBatchDelta ) std::fill( std::begin( vec ), std::end( vec ), 0.0 );

			for( size_t i = begin; i < end; i++ ) {

				const GX_DataVector & currInput = input[ idxOfData[ i ] ];
				const GX_DataVector & currTarget = target[ idxOfData[ i ] ];

				if( ! forwardInternal( currInput, &output ) ) return false;

				if( ! backward( currTarget, output, &delta ) ) return false;

				if( ! collect( currInput, output, delta, &grad ) ) return false;

				GX_Utils::addMatrix( &miniBatchDelta, delta );
				GX_Utils::addMatrix( &miniBatchGrad, grad );

				GX_DataType loss = calcLoss( currTarget, output.back() );

				totalLoss += loss;

				if( mIsDebug )  printf( "DEBUG: input #%ld loss %.8f totalLoss %.8f\n", i, loss, totalLoss );
			}

			if( mIsDebug ) GX_Utils::printMatrix( "batch grad", miniBatchGrad );

			if( ! apply( miniBatchDelta, miniBatchGrad, end - begin,
					learningRate, lambda, input.size() ) ) return false;

			begin += miniBatchCount;
			end = begin + miniBatchCount;
		}

		if( NULL != losses ) ( *losses )[ n ] = totalLoss / input.size();

		if( logInterval <= 1 || ( logInterval > 1 && 0 == n % logInterval ) || n == ( epochCount - 1 ) ) {
			time_t currTime = time( NULL );
			printf( "%s\tinterval %ld [>] epoch %d, lr %f, loss %.8f\n",
				ctime( &currTime ), currTime - beginTime, n, learningRate, totalLoss / input.size() );
			beginTime = time( NULL );
		}

		if( mIsDebug ) print();
	}

	return true;
}

