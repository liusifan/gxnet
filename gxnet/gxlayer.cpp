#include "gxlayer.h"

#include "gxutils.h"
#include "gxact.h"

#include <limits.h>
#include <cstdio>

#include <iostream>

GX_BaseLayer :: GX_BaseLayer( int type )
{
	mIsDebug = false;
	mType = type;
	mActFunc = NULL;
}

GX_BaseLayer :: ~GX_BaseLayer()
{
	if( NULL != mActFunc ) delete mActFunc;
}

void GX_BaseLayer :: forward( const GX_DataVector & input, GX_DataVector * output ) const
{
	assert( input.size() == getInputSize() );

	calcOutput( input, output );
	if( NULL != mActFunc ) mActFunc->activate( *output, output );
}

void GX_BaseLayer :: backward( const GX_DataVector & input, const GX_DataVector & output,
		GX_DataVector * outDelta, GX_DataVector * inDelta ) const
{
	assert( output.size() == outDelta->size() );

	if( NULL != mActFunc ) mActFunc->derivate( output, outDelta );

	if( NULL != inDelta ) backpropagate( input, output, *outDelta, inDelta );
}

const size_t GX_BaseLayer :: getInputSize() const
{
	return gx_dims_flatten_size( mInputDims );
}

const GX_Dims & GX_BaseLayer :: getInputDims() const
{
	return mInputDims;
}

const size_t GX_BaseLayer :: getOutputSize() const
{
	return gx_dims_flatten_size( mOutputDims );
}

const GX_Dims & GX_BaseLayer :: getOutputDims() const
{
	return mOutputDims;
}

int GX_BaseLayer :: getType() const
{
	return mType;
}

void GX_BaseLayer :: setActFunc( GX_ActFunc * actFunc )
{
	if( NULL != mActFunc ) delete mActFunc;
	mActFunc = actFunc;
}

const GX_ActFunc * GX_BaseLayer :: getActFunc() const
{
	return mActFunc;
}

void GX_BaseLayer :: setDebug( bool isDebug )
{
	mIsDebug = isDebug;
}

void GX_BaseLayer :: print( bool isDetail ) const
{
	printf( "Type = %d; ActFuncType = %d; InputDims = %s; OutputDims = %s;\n",
			mType, mActFunc ? mActFunc->getType() : -1,
			gx_vector2string( mInputDims ).c_str(),
			gx_vector2string( mOutputDims ).c_str() );

	printWeights( isDetail );
}

void GX_BaseLayer :: initGradientMatrix( GX_DataMatrix * gradient ) const
{
	/* do nothing */
}

void GX_BaseLayer :: collectGradient( const GX_DataVector & input, const GX_DataVector & output,
		const GX_DataVector & delta, GX_DataMatrix::iterator * iter ) const
{
	/* do nothing */
}

void GX_BaseLayer :: applyGradient( const GX_DataVector & delta, GX_DataMatrix::const_iterator * iter,
		size_t miniBatchCount, GX_DataType learningRate, GX_DataType lambda, size_t trainingCount )
{
	/* do nothing */
}

////////////////////////////////////////////////////////////

GX_ConvLayer :: GX_ConvLayer( const GX_Dims & inputDims, size_t filterCount, size_t filterSize )
	: GX_BaseLayer( GX_BaseLayer::eConv )
{
	assert( inputDims.size() == 3 );

	mFilterDims = { filterCount, inputDims[ 0 ], filterSize, filterSize };

	mFilters.resize( gx_dims_flatten_size( mFilterDims ) );
	for( auto & item : mFilters ) item = GX_Utils::random();

	mBiases.resize( filterCount );
	for( auto & item : mBiases ) item = GX_Utils::random();

	mInputDims = inputDims;
	mOutputDims = {
		filterCount,
		mInputDims[ 1 ] - filterSize + 1,
		mInputDims[ 2 ] - filterSize + 1
	};
}

GX_ConvLayer :: GX_ConvLayer( const GX_Dims & inputDims, const GX_DataVector & filters,
		const GX_Dims & filterDims, const GX_DataVector & biases )
	: GX_BaseLayer( GX_BaseLayer::eConv )
{
	mFilters = filters;

	mFilterDims = filterDims;

	mInputDims = inputDims;
	mOutputDims = {
		mFilterDims[ 0 ],
		mInputDims[ 1 ] - mFilterDims[ 2 ] + 1,
		mInputDims[ 2 ] - mFilterDims[ 3 ] + 1
	};

	mBiases = biases;

	assert( mInputDims[ 0 ] == filterDims[ 1 ] );
}

GX_ConvLayer :: ~GX_ConvLayer()
{
}

void GX_ConvLayer :: printWeights( bool isDetail ) const
{
	printf( "\nfilterDims = %s\n", gx_vector2string( mFilterDims ).c_str() );

	if( !isDetail ) return;

	GX_Utils::printVector( "filters", mFilters, mFilterDims, false );
	GX_Utils::printVector( "biases", mBiases, false );
}

const GX_Dims & GX_ConvLayer :: getFilterDims() const
{
	return mFilterDims;
}

const GX_DataVector & GX_ConvLayer :: getFilters() const
{
	return mFilters;
}

const GX_DataVector & GX_ConvLayer :: getBiases() const
{
	return mBiases;
}

void GX_ConvLayer :: calcOutput( const GX_DataVector & input, GX_DataVector * output ) const
{
	if( output->size() == 0 ) output->resize( gx_dims_flatten_size( mOutputDims ) );

	GX_MDSpanRO inMS( input, mInputDims );

	GX_MDSpanRW outMS( *output, mOutputDims );

	GX_MDSpanRO filterMS( mFilters, mFilterDims );

	for( size_t f = 0; f < mFilterDims[ 0 ]; f++ ) {
		for( size_t x = 0; x < mOutputDims[ 1 ]; x++ ) {
			for( size_t y = 0; y < mOutputDims[ 2 ]; y++ ) {
				outMS( f, x, y ) = forwardConv( inMS, f, x, y, filterMS ) + mBiases[ f ];
			}
		}
	}
}

GX_DataType GX_ConvLayer :: forwardConv( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY,
		GX_MDSpanRO & filterMS )
{
	GX_DataType total = 0;

	for( size_t c = 0; c < filterMS.dim( 1 ); c++ ) {
		for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
				total += inMS( c, beginX + x, beginY + y ) * filterMS( filterIndex, c, x, y );
			}
		}
	}

	return total;
}

void GX_ConvLayer :: backpropagate( const GX_DataVector & input, const GX_DataVector & output,
		const GX_DataVector & outDelta, GX_DataVector * inDelta ) const
{
	// 1. prepare outDelta padding data
	GX_Dims outPaddingDims = {
			mOutputDims[ 0 ],
			mOutputDims[ 1 ] + 2 * ( mFilterDims[ 2 ] - 1 ),
			mOutputDims[ 2 ] + 2 * ( mFilterDims[ 3 ] - 1 )
	};

	GX_DataVector outPadding( gx_dims_flatten_size( outPaddingDims ) );

	GX_MDSpanRW outPaddingMS( outPadding, outPaddingDims );
	GX_MDSpanRO outDeltaMS( outDelta, mOutputDims );

	copyOutDelta( outDeltaMS, mFilterDims[ 2 ], &outPaddingMS );
	if( mIsDebug ) GX_Utils::printVector( "outPadding", outPadding, outPaddingDims, false );

	// 2. prepare rotate180 filters
	GX_DataVector rot180Filters( mFilters.size() );
	rotate180Filter( mFilters, mFilterDims, &rot180Filters );
	if( mIsDebug ) GX_Utils::printVector( "rot180Filters", rot180Filters, mFilterDims, false );

	// 3. convolution
	GX_MDSpanRW inDeltaMS( *inDelta, mInputDims );
	GX_MDSpanRO rot180FiltersMS( rot180Filters, mFilterDims );
	GX_MDSpanRO outPaddingRO( outPadding, outPaddingDims );

	for( size_t c = 0; c < mInputDims[ 0 ]; c++ ) {
		for( size_t x = 0; x < mInputDims[ 1 ]; x++ ) {
			for( size_t y = 0; y < mInputDims[ 2 ]; y++ ) {
				inDeltaMS( c, x, y ) = backwardConv( outPaddingRO, c, x, y, rot180FiltersMS );
			}
		}
	}
}

GX_DataType GX_ConvLayer :: backwardConv( GX_MDSpanRO & inMS, size_t channelIndex, size_t beginX, size_t beginY,
		GX_MDSpanRO & filterMS )
{
	GX_DataType total = 0;

	for( size_t f = 0; f < filterMS.dim( 0 ); f++ ) {
		for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
				total += inMS( f, beginX + x, beginY + y ) * filterMS( f, channelIndex, x, y );
			}
		}
	}

	return total;
}

void GX_ConvLayer :: copyOutDelta( const GX_MDSpanRO & outDeltaMS, size_t filterSize, GX_MDSpanRW * outPaddingMS )
{
	for( size_t f = 0; f < outDeltaMS.dim( 0 ); f++ ) {
		for( size_t x = 0; x < outDeltaMS.dim( 1 ); x++ ) {
			for( size_t y = 0; y < outDeltaMS.dim( 2 ); y++ ) {
				( *outPaddingMS )( f, x + filterSize - 1, y + filterSize - 1 ) = outDeltaMS( f, x, y );
			}
		}
	}
}

void GX_ConvLayer :: rotate180Filter( const GX_DataVector & src, const GX_Dims & dims, GX_DataVector * dest )
{
	GX_MDSpanRO srcMS( src, dims );
	GX_MDSpanRW destMS( *dest, dims );

	dest->resize( gx_dims_flatten_size( dims ) );

	for( size_t f = 0; f < dims[ 0 ]; f++ ) {
		for( size_t c = 0; c < dims[ 1 ]; c++ ) {
			for( size_t i = 0; i < dims[ 2 ]; i++ ) {
				for( size_t j = 0; j < dims[ 3 ]; j++ ) {
					destMS( f, c, dims[ 2 ] - i - 1, dims[ 3 ] - j - 1 ) = srcMS( f, c, i, j );
				}
			}
		}
	}
}

void GX_ConvLayer :: initGradientMatrix( GX_DataMatrix * gradient ) const
{
	gradient->emplace_back( GX_DataVector( gx_dims_flatten_size( mFilterDims ) ) );
}

void GX_ConvLayer :: collectGradient( const GX_DataVector & input, const GX_DataVector & output,
		const GX_DataVector & delta, GX_DataMatrix::iterator * iter ) const
{
	GX_MDSpanRO inMS( input, mInputDims );
	GX_MDSpanRO deltaMS( delta, mOutputDims );

	GX_MDSpanRW gradientMS( *( *iter ), mFilterDims );

	for( size_t f = 0; f < mOutputDims[ 0 ]; f++ ) {
		for( size_t c = 0; c < mInputDims[ 0 ]; c++ ) {
			for( size_t x = 0; x < mFilterDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < mFilterDims[ 3 ]; y++ ) {
					gradientMS( f, c, x, y ) = gradientConv( inMS, f, c, x, y, deltaMS );
				}
			}
		}
	}

	( *iter )++;
}

GX_DataType GX_ConvLayer :: gradientConv( GX_MDSpanRO & inMS, size_t filterIndex, size_t channelIndex,
		size_t beginX, size_t beginY, GX_MDSpanRO & filterMS )
{
	GX_DataType total = 0;

	for( size_t x = 0; x < filterMS.dim( 1 ); x++ ) {
		for( size_t y = 0; y < filterMS.dim( 2 ); y++ ) {
			total += inMS( channelIndex, beginX + x, beginY + y ) * filterMS( filterIndex, x, y );
		}
	}

	return total;
}

void GX_ConvLayer :: applyGradient( const GX_DataVector & delta, GX_DataMatrix::const_iterator * iter,
		size_t miniBatchCount, GX_DataType learningRate, GX_DataType lambda, size_t trainingCount )
{
	GX_MDSpanRO gradientMS( *( *iter ), mFilterDims );

	GX_MDSpanRW filtersMS( mFilters, mFilterDims );

	for( size_t f = 0; f < mFilterDims[ 0 ]; f++ ) {
		for( size_t c = 0; c < mFilterDims[ 1 ]; c++ ) {
			for( size_t i = 0; i < mFilterDims[ 2 ]; i++ ) {
				for( size_t j = 0; j < mFilterDims[ 3 ]; j++ ) {
					if( mIsDebug ) {
						filtersMS( f, c, i, j ) = filtersMS( f, c, i, j ) - gradientMS( f, c, i, j ) * learningRate;
					} else {
						filtersMS( f, c, i, j ) = ( 1 - learningRate * lambda / trainingCount ) * filtersMS( f, c, i, j )
								- gradientMS( f, c, i, j ) * learningRate / miniBatchCount;
					}
				}
			}
		}
	}

	( *iter )++;

	GX_MDSpanRO deltaMS( delta, mOutputDims );
	for( size_t f = 0; f < mOutputDims[ 0 ]; f++ ) {
		GX_DataType biasGradient = 0;
		for( size_t i = 0; i < mOutputDims[ 1 ]; i++ ) {
			for( size_t j = 0; j < mOutputDims[ 2 ]; j++ ) {
				biasGradient += deltaMS( f, i, j );
			}
		}
		if( mIsDebug ) printf( "bias#%zu.gradient %f\n", f, biasGradient );
		mBiases[ f ] = mBiases[ f ] - biasGradient * learningRate / miniBatchCount;
	}
}

////////////////////////////////////////////////////////////

GX_MaxPoolLayer :: GX_MaxPoolLayer( const GX_Dims & inputDims, size_t poolSize )
	: GX_BaseLayer( GX_BaseLayer::eMaxPool )
{
	assert( inputDims.size() == 3 );

	mInputDims = inputDims;
	mOutputDims = { mInputDims[ 0 ], mInputDims[ 1 ] / poolSize, mInputDims[ 2 ] / poolSize };

	mPoolSize = poolSize;
}

GX_MaxPoolLayer :: ~GX_MaxPoolLayer()
{
}

void GX_MaxPoolLayer :: printWeights( bool isDetail ) const
{
	printf( "\nPoolSize = %zu\n", mPoolSize );
}

size_t GX_MaxPoolLayer :: getPoolSize() const
{
	return mPoolSize;
}

void GX_MaxPoolLayer :: calcOutput( const GX_DataVector & input, GX_DataVector * output ) const
{
	if( output->size() == 0 ) output->resize( gx_dims_flatten_size( mOutputDims ) );

	GX_MDSpanRO inMS( input, mInputDims );

	GX_MDSpanRW outMS( *output, mOutputDims );

	for( size_t f = 0; f < mOutputDims[ 0 ]; f++ ) {
		for( size_t x = 0; x < mOutputDims[ 1 ]; x++ ) {
			for( size_t y = 0; y < mOutputDims[ 2 ]; y++ ) {
				outMS( f, x, y ) = pool( inMS, f, x * mPoolSize, y * mPoolSize );
			}
		}
	}
}

GX_DataType GX_MaxPoolLayer :: pool( GX_MDSpanRO & inMS, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	GX_DataType result = 1.0 * INT_MIN;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result = std::max( result, inMS( filterIndex, beginX + x, beginY + y ) );
		}
	}

	return result;
}

void GX_MaxPoolLayer :: backpropagate( const GX_DataVector & input, const GX_DataVector & output,
		const GX_DataVector & outDelta, GX_DataVector * inDelta ) const
{
	*inDelta = 0;
	GX_MDSpanRW inDeltaMS( *inDelta, mInputDims );

	GX_MDSpanRO outDeltaMS( outDelta, mOutputDims );

	GX_MDSpanRO inMS( input, mInputDims );
	GX_MDSpanRO outputMS( output, mOutputDims );

	for( size_t f = 0; f < mOutputDims[ 0 ]; f++ ) {
		for( size_t x = 0; x < mOutputDims[ 1 ]; x++ ) {
			for( size_t y = 0; y < mOutputDims[ 2 ]; y++ ) {
				unpool( inMS, f, x * mPoolSize, y * mPoolSize,
						outputMS( f, x, y ), outDeltaMS( f, x, y ), &inDeltaMS );
			}
		}
	}
}

void GX_MaxPoolLayer :: unpool( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY,
		const GX_DataType maxValue, GX_DataType outDelta, GX_MDSpanRW * inDeltaMS ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			if( inMS( filterIndex, beginX + x, beginY + y ) == maxValue ) {
				( *inDeltaMS )( filterIndex, beginX + x, beginY + y ) = outDelta;
			}
		}
	}
}

////////////////////////////////////////////////////////////

GX_AvgPoolLayer :: GX_AvgPoolLayer( const GX_Dims & inputDims, size_t poolSize )
	: GX_BaseLayer( GX_BaseLayer::eAvgPool )
{
	assert( inputDims.size() == 3 );

	mInputDims = inputDims;
	mOutputDims = { mInputDims[ 0 ], mInputDims[ 1 ] / poolSize, mInputDims[ 2 ] / poolSize };

	mPoolSize = poolSize;
}

GX_AvgPoolLayer :: ~GX_AvgPoolLayer()
{
}

void GX_AvgPoolLayer :: printWeights( bool isDetail ) const
{
	printf( "\nPoolSize = %zu\n", mPoolSize );
}

size_t GX_AvgPoolLayer :: getPoolSize() const
{
	return mPoolSize;
}

void GX_AvgPoolLayer :: calcOutput( const GX_DataVector & input, GX_DataVector * output ) const
{
	if( output->size() == 0 ) output->resize( gx_dims_flatten_size( mOutputDims ) );

	GX_MDSpanRO inMS( input, mInputDims );

	GX_MDSpanRW outMS( *output, mOutputDims );

	for( size_t f = 0; f < mOutputDims[ 0 ]; f++ ) {
		for( size_t x = 0; x < mOutputDims[ 1 ]; x++ ) {
			for( size_t y = 0; y < mOutputDims[ 2 ]; y++ ) {
				outMS( f, x, y ) = pool( inMS, f, x * mPoolSize, y * mPoolSize );
			}
		}
	}
}

GX_DataType GX_AvgPoolLayer :: pool( GX_MDSpanRO & inMS, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	GX_DataType result = 0;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result += inMS( filterIndex, beginX + x, beginY + y );
		}
	}

	return result / ( mPoolSize * mPoolSize );
}

void GX_AvgPoolLayer :: backpropagate( const GX_DataVector & input, const GX_DataVector & output,
		const GX_DataVector & outDelta, GX_DataVector * inDelta ) const
{
	*inDelta = 0;
	GX_MDSpanRW inDeltaMS( *inDelta, mInputDims );

	GX_MDSpanRO outDeltaMS( outDelta, mOutputDims );

	GX_MDSpanRO inMS( input, mInputDims );
	GX_MDSpanRO outputMS( output, mOutputDims );

	for( size_t f = 0; f < mOutputDims[ 0 ]; f++ ) {
		for( size_t x = 0; x < mOutputDims[ 1 ]; x++ ) {
			for( size_t y = 0; y < mOutputDims[ 2 ]; y++ ) {
				unpool( inMS, f, x * mPoolSize, y * mPoolSize,
						outputMS( f, x, y ), outDeltaMS( f, x, y ), &inDeltaMS );
			}
		}
	}
}

void GX_AvgPoolLayer :: unpool( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY,
		const GX_DataType maxValue, GX_DataType outDelta, GX_MDSpanRW * inDeltaMS ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			( *inDeltaMS )( filterIndex, beginX + x, beginY + y ) = outDelta / ( mPoolSize * mPoolSize );
		}
	}
}

////////////////////////////////////////////////////////////

GX_FullConnLayer :: GX_FullConnLayer( size_t neuronCount, size_t inputCount )
	: GX_BaseLayer( GX_BaseLayer::eFullConn )
{
	mWeights.resize( neuronCount );
	for( auto & neuron : mWeights ) {
		neuron.resize( inputCount );
		for( auto & w : neuron ) w = GX_Utils::random();
	}

	mBiases.resize( neuronCount );
	for( auto & b : mBiases ) b = GX_Utils::random();

	mInputDims = { inputCount };
	mOutputDims = { neuronCount };
}

GX_FullConnLayer :: ~GX_FullConnLayer()
{
}

void GX_FullConnLayer :: printWeights( bool isDetail ) const
{
	if( !isDetail ) return;

	printf( "Weights: Count = %zu; InputCount = %zu;\n", mWeights.size(), mWeights[ 0 ].size() );
	for( size_t i = 0; i < mWeights.size() && i < 10; i++ ) {
		printf( "\tNeuron#%zu: WeightCount = %zu, Bias = %.8f\n", i, mWeights[ i ].size(), mBiases[ i ] );
		for( size_t j = 0; j < mWeights[ i ].size() && j < 10; j++ ) {
			printf( "\t\tWeight#%zu: %.8f\n", j, mWeights[ i ][ j ] );
		}

		if( mWeights[ i ].size() > 10 ) printf( "\t\t......\n" );
	}

	if( mWeights.size() > 10 ) printf( "\t......\n" );
}

const GX_DataMatrix & GX_FullConnLayer :: getWeights() const
{
	return mWeights;
}

const GX_DataVector & GX_FullConnLayer :: getBiases() const
{
	return mBiases;
}

void GX_FullConnLayer :: setWeights( const GX_DataMatrix & weights, const GX_DataVector & biases )
{
	mWeights = weights;
	mBiases = biases;
}

void GX_FullConnLayer :: calcOutput( const GX_DataVector & input, GX_DataVector * output ) const
{
	if( output->size() == 0 ) output->resize( gx_dims_flatten_size( mOutputDims ) );

	assert( output->size() == mWeights.size() );

	for( size_t i = 0; i < mWeights.size(); i++ ) {
		( *output )[ i ]  = ( mWeights[ i ] * input ).sum();
		if( ! mIsDebug )  ( *output )[ i ] += mBiases[ i ];
	}
}

void GX_FullConnLayer :: backpropagate( const GX_DataVector & /* unused */, const GX_DataVector & output,
		const GX_DataVector & outDelta, GX_DataVector * inDelta ) const
{
	if( NULL != inDelta ) {
		( *inDelta ) = 0;
		for( size_t i = 0; i < inDelta->size(); i++ ) {
			for( size_t j = 0; j < outDelta.size(); j++ ) {
				( *inDelta )[ i ] += outDelta[ j ] * mWeights[ j ][ i ];
			}
		}
	}
}

void GX_FullConnLayer :: initGradientMatrix( GX_DataMatrix * gradient ) const
{
	for( size_t i = 0; i < getOutputSize(); i++ ) {
		gradient->emplace_back( GX_DataVector( getInputSize() ) );
	}
}

void GX_FullConnLayer :: collectGradient( const GX_DataVector & input, const GX_DataVector & output,
		const GX_DataVector & delta, GX_DataMatrix::iterator * gradient ) const
{
	for( size_t i = 0; i < mWeights.size(); i++ ) {
		for( size_t j = 0; j < mWeights[ 0 ].size(); j++ ) {
			( *( *gradient ) )[ j ] = delta[ i ] * input[ j ];
		}

		++( *gradient );
	}
}

void GX_FullConnLayer :: applyGradient( const GX_DataVector & delta, GX_DataMatrix::const_iterator * iter,
		size_t miniBatchCount, GX_DataType learningRate, GX_DataType lambda, size_t trainingCount )
{
	for( size_t i = 0; i < mWeights.size(); i++ ) {
		GX_DataVector & weights = mWeights[ i ];
		for( size_t j = 0; j < weights.size(); j++ ) {
			if( mIsDebug ) {
				weights[ j ] = weights[ j ] - ( *( *iter ) )[ j ] * learningRate;
			} else {
				weights[ j ] = ( 1 - learningRate * lambda / trainingCount ) * weights[ j ] 
						- ( *( *iter ) )[ j ] * learningRate / miniBatchCount;
			}
		}
		( *iter )++;
	}

	if( ! mIsDebug ) mBiases -= learningRate * delta / miniBatchCount;
}

