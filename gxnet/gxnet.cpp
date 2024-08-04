
#include "gxnet.h"
#include "gxutils.h"

#include <random>
#include <numeric>
#include <algorithm>

#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>

GX_Network :: GX_Network( int lossFuncType )
{
	mOnEpochEnd = NULL;
	mLossFuncType = lossFuncType;
	mIsDebug = false;
	mIsShuffle = true;
}

GX_Network :: ~GX_Network()
{
	for( auto & item : mLayers ) delete item;
}

void GX_Network :: print( bool isDetail ) const
{
	printf( "\n{{{ isDetail %s\n", isDetail ? "true" : "false" );
	printf( "Network: LayerCount = %zu; LossFuncType = %d;\n", mLayers.size(), mLossFuncType );
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_BaseLayer * layer = mLayers[ i ];

		printf( "\nLayer#%ld: ", i );
		layer->print( isDetail );
	}
	printf( "}}}\n\n" );
}

void GX_Network :: setOnEpochEnd( GX_OnEpochEnd_t onEpochEnd )
{
	mOnEpochEnd = onEpochEnd;
}

void GX_Network :: setDebug( bool isDebug )
{
	mIsDebug = isDebug;

	for( auto & layer : mLayers ) layer->setDebug( isDebug );
}

void GX_Network :: setShuffle( bool isShuffle )
{
	mIsShuffle = isShuffle;
}

void GX_Network :: setLossFuncType( int lossFuncType )
{
	mLossFuncType = lossFuncType;
}

int GX_Network :: getLossFuncType() const
{
	return mLossFuncType;
}

const GX_BaseLayer * GX_Network :: lastLayer()
{
	return mLayers.size() > 0 ? mLayers.back() : NULL;
}

GX_BaseLayerPtrVector & GX_Network :: getLayers()
{
	return mLayers;
}

const GX_BaseLayerPtrVector & GX_Network :: getLayers() const
{
	return mLayers;
}

void GX_Network :: addLayer( GX_BaseLayer * layer )
{
	layer->setDebug( mIsDebug );

	mLayers.emplace_back( layer );
}

bool GX_Network :: forward( const GX_DataVector & input, GX_DataMatrix * output ) const
{
	if( input.size() != mLayers[ 0 ]->getInputSize() ) {
		printf( "%s input.size %zu, layer[0].inputSize %zu",
				__func__, input.size(), mLayers[ 0 ]->getInputSize() );
		return false;
	}

	if( output->size() == 0 ) {
		for( auto & item : mLayers ) output->emplace_back( GX_DataVector( item->getOutputSize() ) );
	}

	const GX_DataVector * currInput = &input;

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_BaseLayer * layer = mLayers[ i ];

		if( i > 0 ) currInput = &( (*output)[ i - 1 ] );

		layer->forward( *currInput, &( ( *output )[ i ] ) );
	}

	return true;
}

bool GX_Network :: apply( const GX_DataMatrix & delta, const GX_DataMatrix & gradient,
		int miniBatchCount, GX_DataType learningRate, GX_DataType lambda, int trainingCount )
{
	GX_DataMatrix::const_iterator iter = gradient.begin();

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		GX_BaseLayer * layer = mLayers[ i ];
		layer->applyGradient( delta[ i ], &iter, miniBatchCount, learningRate, lambda, trainingCount );
	}

	return true;
}

void GX_Network :: collect( const GX_DataVector & input, const GX_DataMatrix & output,
			const GX_DataMatrix & delta, GX_DataMatrix * gradient )
{
	const GX_DataVector * currInput = &input;

	GX_DataMatrix::iterator iter = gradient->begin();

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		if( i > 0 ) currInput = &( output[ i - 1 ] );

		GX_BaseLayer * layer = mLayers[ i ];

		layer->collectGradient( ( *currInput ), output[ i ], delta[ i ], &iter );
	}

	if( mIsDebug ) {
		GX_Utils::printVector( "input", input );
		GX_Utils::printMatrix( "output", output );
		GX_Utils::printMatrix( "delta", delta );
		GX_Utils::printMatrix( "gradient", *gradient );
	}
}

bool GX_Network :: backward( const GX_DataVector & input, const GX_DataVector & target,
		const GX_DataMatrix & output, GX_DataMatrix * delta )
{
	const GX_DataVector & lastOutput = output.back();

	if( eMeanSquaredError == mLossFuncType ) {
		delta->back() = 2.0 * ( lastOutput - target );
	}

	if( eCrossEntropy == mLossFuncType ) {
		delta->back() = lastOutput - target;
	}

	for( ssize_t i = mLayers.size() - 1; i >= 0; i-- ) {
		GX_DataVector * inDelta = ( i > 0 ) ? &( ( *delta )[ i - 1 ] ) : NULL;

		const GX_DataVector & currInput = ( i > 0 ) ? ( output[ i - 1 ] ) : input;

		GX_BaseLayer * layer = mLayers[ i  ];
		layer->backward( currInput, output[ i ], &( ( *delta ) [ i ] ), inDelta );
	}

	return true;
}

void GX_Network :: initGradientMatrix( GX_DataMatrix * batchGradient, GX_DataMatrix * gradient )
{
	size_t total = 0;
	for( auto & layer : mLayers ) total += layer->getOutputSize();

	batchGradient->reserve( total );
	gradient->reserve( total );

	for( auto & layer : mLayers ) {
		layer->initGradientMatrix( gradient );
		layer->initGradientMatrix( batchGradient );
	}
}

void GX_Network :: initOutputAndDeltaMatrix( GX_DataMatrix * output, GX_DataMatrix * batchDelta, GX_DataMatrix * delta )
{
	for( auto & item : mLayers ) output->emplace_back( GX_DataVector( item->getOutputSize() ) );

	for( auto & item : mLayers ) delta->emplace_back( GX_DataVector( item->getOutputSize() ) );

	for( auto & item : *delta ) batchDelta->emplace_back( GX_DataVector( item.size() ) );
}

GX_DataType GX_Network :: calcLoss( const GX_DataVector & target, const GX_DataVector & output )
{
	GX_DataType ret = 0;

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

bool GX_Network :: trainInternal( const GX_DataMatrix & input, const GX_DataMatrix & target, int epochCount,
		int miniBatchCount, GX_DataType learningRate, GX_DataType lambda, GX_DataVector * losses )
{
	if( input.size() != target.size() ) return false;

	time_t beginTime = time( NULL );

	printf( "%s\tstart train, input { %zu }, target { %zu }\n",
			ctime( &beginTime ), input.size(), target.size() );

	int logInterval = epochCount / 10;
	int progressInterval = ( input.size() / miniBatchCount ) / 10;

	std::random_device rd;
	std::mt19937 gen( rd() );

	assert( mLayers[ 0 ]->getInputSize() == input[ 0 ].size() );

	GX_DataMatrix batchGradient, gradient;
	initGradientMatrix( &batchGradient, &gradient );

	GX_DataMatrix output, batchDelta, delta;
	initOutputAndDeltaMatrix( &output, &batchDelta, &delta );

	if( NULL != losses ) losses->resize( epochCount, 0 );

	for( int n = 0; n < epochCount; n++ ) {

		std::vector< int > idxOfData( input.size() );
		std::iota( idxOfData.begin(), idxOfData.end(), 0 );
		if( mIsShuffle ) std::shuffle( idxOfData.begin(), idxOfData.end(), gen );

		GX_DataType totalLoss = 0;

		miniBatchCount = std::max( miniBatchCount, 1 );

		for( size_t begin = 0; begin < idxOfData.size(); ) {
			size_t end = std::min( idxOfData.size(), begin + miniBatchCount );

			for( auto & vec : batchGradient ) std::fill( std::begin( vec ), std::end( vec ), 0.0 );
			for( auto & vec : batchDelta ) std::fill( std::begin( vec ), std::end( vec ), 0.0 );

			for( size_t i = begin; i < end; i++ ) {

				const GX_DataVector & currInput = input[ idxOfData[ i ] ];
				const GX_DataVector & currTarget = target[ idxOfData[ i ] ];

				forward( currInput, &output );

				backward( currInput, currTarget, output, &delta );

				collect( currInput, output, delta, &gradient );

				gx_add_matrix( &batchDelta, delta );
				gx_add_matrix( &batchGradient, gradient );

				GX_DataType loss = calcLoss( currTarget, output.back() );

				totalLoss += loss;

				if( mIsDebug )  printf( "DEBUG: input #%ld loss %.8f totalLoss %.8f\n", i, loss, totalLoss );
			}

			if( mIsDebug ) {
				GX_Utils::printMatrix( "batch delta", batchDelta );
				GX_Utils::printMatrix( "batch gradient", batchGradient );
			}

			apply( batchDelta, batchGradient, end - begin, learningRate, lambda, input.size() );

			begin += miniBatchCount;
			end = begin + miniBatchCount;

			if( progressInterval > 0 && 0 == ( begin % ( progressInterval * miniBatchCount ) ) ) {
				printf( "\r%zu / %zu", begin, idxOfData.size() );
				fflush( stdout );
			}
		}

		if( NULL != losses ) ( *losses )[ n ] = totalLoss / input.size();

		if( logInterval <= 1 || ( logInterval > 1 && 0 == n % logInterval ) || n == ( epochCount - 1 ) ) {
			time_t currTime = time( NULL );
			printf( "\r%s\tinterval %ld [>] epoch %d, lr %f, loss %.8f\n",
				ctime( &currTime ), currTime - beginTime, n, learningRate, totalLoss / input.size() );
			beginTime = time( NULL );
		}

		if( mIsDebug ) print();

		if( mOnEpochEnd ) mOnEpochEnd( *this, n, totalLoss / input.size() );
	}

	return true;
}

bool GX_Network :: train( const GX_DataMatrix & input, const GX_DataMatrix & target, int epochCount,
		int miniBatchCount, GX_DataType learningRate, GX_DataType lambda, GX_DataVector * losses )
{
	std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();	

	bool ret = trainInternal( input, target, epochCount, miniBatchCount, learningRate, lambda, losses );

	std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();	

	auto timeSpan = std::chrono::duration_cast<std::chrono::milliseconds>( endTime - beginTime );

	printf( "Elapsed time: %.3f\n", timeSpan.count() / 1000.0 );

	return ret;
}

