#pragma once

#include "gxcomm.h"
#include "gxlayer.h"

class GX_Network {
public:
	enum { eNone, eMeanSquaredError, eCrossEntropy };

	GX_Network( int lossFuncType = eMeanSquaredError );
	~GX_Network();

	void setDebug( bool isDebug );

	void setShuffle( bool isShuffle );

	void setLossFuncType( int lossFuncType );

	int getLossFuncType() const;

	void addLayer( GX_BaseLayer * layer );

	const GX_BaseLayer * lastLayer();

	GX_BaseLayerPtrVector & getLayers();

	const GX_BaseLayerPtrVector & getLayers() const;

	bool forward( const GX_DataVector & input, GX_DataMatrix * output ) const;

	bool backward( const GX_DataVector & input, const GX_DataVector & target,
			const GX_DataMatrix & output, GX_DataMatrix * delta );

	bool train( const GX_DataMatrix & input, const GX_DataMatrix & target, int epochCount,
			int miniBatchCount, GX_DataType learningRate, GX_DataType lambda = 0,
			GX_DataVector * losses = nullptr );

	void print( bool isFull = false ) const;

private:

	void collect( const GX_DataVector & input, const GX_DataMatrix & output,
			const GX_DataMatrix & delta, GX_DataMatrix * gradient );

	bool apply( const GX_DataMatrix & delta, const GX_DataMatrix & gradient,
			int miniBatchCount, GX_DataType learningRate,
			GX_DataType lambda, int trainingCount );

	GX_DataType calcLoss( const GX_DataVector & target, const GX_DataVector & output );

	void initGradientMatrix( GX_DataMatrix * batchGradient, GX_DataMatrix * gradient );

	void initOutputAndDeltaMatrix( GX_DataMatrix * output, GX_DataMatrix * batchDelta, GX_DataMatrix * delta );

private:
	int mLossFuncType;
	GX_BaseLayerPtrVector mLayers;
	bool mIsDebug, mIsShuffle;
};

