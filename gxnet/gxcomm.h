#pragma once

#include <vector>
#include <valarray>
#include <string>
#include <sstream>
#include <iostream>

#include <assert.h>

typedef double GX_DataType;

typedef std::valarray< GX_DataType > GX_DataVector;

typedef std::vector< GX_DataVector > GX_DataMatrix;

typedef std::vector< std::string > GX_StringList;

typedef std::vector< size_t > GX_Dims;
typedef std::vector< GX_Dims > GX_DimsList;

inline size_t gx_dims_flatten_size( const GX_Dims & dims )
{
	size_t ret = dims.size() > 0 ? 1 : 0;
	for( auto & dim : dims ) ret = ret * dim;

	return ret;
}

template< typename NumberVector >
std::string gx_vector2string( const NumberVector & vec, const char delim = ',' )
{
	std::ostringstream ret;
	ret.setf( std::ios::scientific, std::ios::floatfield );

	for( size_t i = 0; i < vec.size(); i++ ) {
		if( i > 0 ) ret << delim;
		ret << vec[ i ];
	}

	return ret.str();
}

template< typename NumberVector >
void gx_string2vector( const std::string & buff, NumberVector * vec, const char delim = ',' )
{
	std::stringstream ss( buff );
	std::string token;
	while( std::getline( ss, token, delim ) ) {
		vec->push_back( std::stod( token ) );
	}
}

inline void gx_string2valarray( const std::string & buff, GX_DataVector * vec, const char delim = ',' )
{
	std::stringstream ss( buff );
	std::string token;
	for( size_t i = 0; i < vec->size(); i++ ) {
		if( !std::getline( ss, token, delim ) ) break;
		( *vec )[ i ] = std::stod( token );
	}
}

class GX_MDSpanRO {
public:
	GX_MDSpanRO( const GX_DataVector & data, const GX_Dims & dims )
			: mData( data ), mDims( dims ) {
		assert( gx_dims_flatten_size( dims ) == data.size() );
	}

	~GX_MDSpanRO() {}

	const GX_Dims & dims() const { return mDims; }

	size_t dim( size_t index ) const {
		assert( index < mDims.size() );
		return mDims[ index ];
	}

	const GX_DataType & operator()( size_t i ) const {
		return mData[ i ];
	}

	const GX_DataType & operator()( size_t i, size_t j ) const {
		assert( mDims.size() == 2 );
		return mData[ i * mDims[ 1 ] + j ];
	}

	const GX_DataType & operator()( size_t i, size_t j, size_t k ) const {
		assert( mDims.size() == 3 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * i ) + ( mDims[ 2 ] * j ) + k ];
	}

	const GX_DataType & operator()( size_t f, size_t c, size_t i, size_t j ) const {
		assert( mDims.size() == 4 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * mDims[ 3 ] * f )
		       + ( mDims[ 2 ] * mDims[ 3 ] * c ) + ( mDims[ 3 ] * i ) + j ];
	}

private:
	const GX_DataVector & mData;
	const GX_Dims & mDims;
};

class GX_MDSpanRW {
public:
	GX_MDSpanRW( GX_DataVector & data, const GX_Dims & dims )
			: mData( data ), mDims( dims ) {
		assert( gx_dims_flatten_size( dims ) == data.size() );
	}

	~GX_MDSpanRW() {}

	const GX_Dims & dims() const { return mDims; }

	size_t dim( size_t index ) const {
		assert( index < mDims.size() );
		return mDims[ index ];
	}

	GX_DataType & operator()( size_t i ) {
		return mData[ i ];
	}

	GX_DataType & operator()( size_t i, size_t j ) {
		assert( mDims.size() == 2 );
		return mData[ i * mDims[ 1 ] + j ];
	}

	GX_DataType & operator()( size_t i, size_t j, size_t k ) {
		assert( mDims.size() == 3 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * i ) + ( mDims[ 2 ] * j ) + k ];
	}

	GX_DataType & operator()( size_t f, size_t c, size_t i, size_t j ) const {
		assert( mDims.size() == 4 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * mDims[ 3 ] * f )
		       + ( mDims[ 2 ] * mDims[ 3 ] * c ) + ( mDims[ 3 ] * i ) + j ];
	}

private:
	GX_DataVector & mData;
	const GX_Dims & mDims;
};

