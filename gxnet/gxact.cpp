
#include "gxact.h"


////////////////////////////////////////////////////////////

GX_ActFunc * GX_ActFunc :: sigmoid()
{
	return new GX_ActFunc( eSigmoid );
}

GX_ActFunc * GX_ActFunc :: tanh()
{
	return new GX_ActFunc( eTanh );
}

GX_ActFunc * GX_ActFunc :: leakyReLU()
{
	return new GX_ActFunc( eLeakyReLU );
}

GX_ActFunc * GX_ActFunc :: softmax()
{
	return new GX_ActFunc( eSoftmax );
}

GX_ActFunc :: GX_ActFunc( int type )
{
	mType = type;
}

GX_ActFunc :: ~GX_ActFunc()
{
}

int GX_ActFunc :: getType() const
{
	return mType;
}

void GX_ActFunc :: activate( const GX_DataVector & input, GX_DataVector * output ) const
{
	if( eSigmoid == mType ) {
		*output = 1.0f / ( 1.0f + std::exp( - input ) );
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < input.size(); i++ ) {
			if( input[ i ] < 0 ) {
				( *output )[ i ] = 0.01 * input[ i ];
			} else if( input[ i ] > 1 ) {
				( *output )[ i ] = 1 + 0.01 * ( input[ i ] - 1 );
			}
		}
	}

	if( eTanh == mType ) {
		*output = std::tanh( input );
	}

	if( eSoftmax == mType ) {
		*output = std::exp( input - input.max() );
		*output /= output->sum();
	}
}

void GX_ActFunc :: derivate( const GX_DataVector & output, GX_DataVector * outDelta ) const
{
	if( eSigmoid == mType ) {
		*outDelta = output * ( 1 - output ) * ( *outDelta );
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			( *outDelta )[ i ] = ( *outDelta )[ i ] * ( output[ i ] < 0 || output[ i ] > 1 ? 0.01 : 1 );
		}
	}

	if( eTanh == mType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			( *outDelta )[ i ] = ( *outDelta )[ i ] * ( 1 - output[ i ] * output[ i ] );
		}
	}

	if( eSoftmax == mType ) {
		GX_DataVector dOutput = *outDelta;
		for( size_t j = 0; j < output.size(); j++ ) {
			GX_DataVector dSoftmax( output.size() );
			for( size_t k = 0; k < output.size(); k++ ) {
				dSoftmax[ k ] = ( k == j ) ? output[ j ] * ( 1.0 - output[ j ] ) : -output[ k ] * output[ j ];
			}

			( *outDelta )[ j ] = ( dOutput * dSoftmax ).sum();
		}
	}
}

