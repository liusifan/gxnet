#pragma once

#include "gxcomm.h"
#include <string>
#include <vector>

class GX_BaseLayer;

typedef std::vector< GX_BaseLayer * > GX_BaseLayerPtrVector;

class GX_ActFunc;

class GX_BaseLayer {
public:
	enum { eNone, eConv, eMaxPool, eAvgPool, eFullConn };

public:
	GX_BaseLayer( int type );

	virtual ~GX_BaseLayer();

	virtual void initGradientMatrix( GX_DataMatrix * gradient ) const;

	virtual void collectGradient( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & delta, GX_DataMatrix::iterator * iter ) const;

	virtual void applyGradient( const GX_DataVector & delta,
			GX_DataMatrix::const_iterator * iter, size_t miniBatchCount,
			GX_DataType learningRate, GX_DataType lambda, size_t trainingCount );

public:

	virtual void print( bool isFull = false ) const;

	void forward( const GX_DataVector & input, GX_DataVector * output ) const;

	void backward( const GX_DataVector & input, const GX_DataVector & output,
			GX_DataVector * outDelta, GX_DataVector * inDelta ) const;

protected:

	virtual void printWeights( bool isFull ) const = 0;

	virtual void calcOutput( const GX_DataVector & input, GX_DataVector * output ) const = 0;

	virtual void backpropagate( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & outDelta, GX_DataVector * inDelta ) const = 0;

public:
	int getType() const;

	const GX_Dims & getInputDims() const;

	const size_t getInputSize() const;

	const GX_Dims & getOutputDims() const;

	const size_t getOutputSize() const;

	void setActFunc( GX_ActFunc * actFunc );

	const GX_ActFunc * getActFunc() const;

	void setDebug( bool isDebug );

protected:
	GX_Dims mInputDims, mOutputDims;
	int mType;
	bool mIsDebug;

	GX_ActFunc * mActFunc;
};

class GX_ConvLayer : public GX_BaseLayer {
public:
	GX_ConvLayer( const GX_Dims & inputDims, size_t filterCount, size_t filterSize );
	GX_ConvLayer( const GX_Dims & inputDims, const GX_DataVector & filters, const GX_Dims & filterDims,
			const GX_DataVector & biases );

	~GX_ConvLayer();

	const GX_Dims & getFilterDims() const;

	const GX_DataVector & getFilters() const;

	const GX_DataVector & getBiases() const;

public:

	virtual void initGradientMatrix( GX_DataMatrix * gradient ) const;

	virtual void collectGradient( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & delta, GX_DataMatrix::iterator * iter ) const;

	virtual void applyGradient( const GX_DataVector & delta,
			GX_DataMatrix::const_iterator * iter, size_t miniBatchCount,
			GX_DataType learningRate, GX_DataType lambda, size_t trainingCount );

	void printWeights( bool isFull ) const;

protected:

	virtual void calcOutput( const GX_DataVector & input, GX_DataVector * output ) const;

	virtual void backpropagate( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & outDelta, GX_DataVector * inDelta ) const;

private:

	static GX_DataType forwardConv( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY,
			GX_MDSpanRO & filterMS );

	static GX_DataType backwardConv( GX_MDSpanRO & inMS, size_t channelIndex, size_t beginX, size_t beginY,
			GX_MDSpanRO & filterMS );

	static GX_DataType gradientConv( GX_MDSpanRO & inMS, size_t filterIndex, size_t channelIndex,
			size_t beginX, size_t beginY, GX_MDSpanRO & filterMS );

	static void rotate180Filter( const GX_DataVector & src, const GX_Dims & dims, GX_DataVector * dest );

	static void copyOutDelta( const GX_MDSpanRO & outDeltaMS, size_t filterSize, GX_MDSpanRW * outPaddingMS );

private:
	GX_Dims mFilterDims;
	GX_DataVector mFilters, mBiases;
};

class GX_MaxPoolLayer : public GX_BaseLayer {
public:
	GX_MaxPoolLayer( const GX_Dims & inputDims, size_t poolSize );
	~GX_MaxPoolLayer();

	virtual void printWeights( bool isFull ) const;

	size_t getPoolSize() const;

protected:

	virtual void calcOutput( const GX_DataVector & input, GX_DataVector * output ) const;

	virtual void backpropagate( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & outDelta, GX_DataVector * inDelta ) const;

private:
	GX_DataType pool( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY ) const;

	void unpool( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY,
			const GX_DataType maxValue, GX_DataType outDelta, GX_MDSpanRW * inDeltaMS ) const;

private:
	size_t mPoolSize;
};

class GX_AvgPoolLayer : public GX_BaseLayer {
public:
	GX_AvgPoolLayer( const GX_Dims & inputDims, size_t poolSize );
	~GX_AvgPoolLayer();

	virtual void printWeights( bool isFull ) const;

	size_t getPoolSize() const;

protected:

	virtual void calcOutput( const GX_DataVector & input, GX_DataVector * output ) const;

	virtual void backpropagate( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & outDelta, GX_DataVector * inDelta ) const;

private:
	GX_DataType pool( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY ) const;

	void unpool( GX_MDSpanRO & inMS, size_t filterIndex, size_t beginX, size_t beginY,
			const GX_DataType maxValue, GX_DataType outDelta, GX_MDSpanRW * inDeltaMS ) const;

private:
	size_t mPoolSize;
};

class GX_FullConnLayer : public GX_BaseLayer {
public:
	GX_FullConnLayer( size_t neuronCount, size_t inputCount );

	~GX_FullConnLayer();

	// for debug
	void setWeights( const GX_DataMatrix & weights, const GX_DataVector & biases );

	void printWeights( bool isFull ) const;

	const GX_DataMatrix & getWeights() const;

	const GX_DataVector & getBiases() const;

	virtual void initGradientMatrix( GX_DataMatrix * gradient ) const;

	virtual void collectGradient( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & delta, GX_DataMatrix::iterator * iter ) const;

	virtual void applyGradient( const GX_DataVector & delta,
			GX_DataMatrix::const_iterator * iter, size_t miniBatchCount,
			GX_DataType learningRate, GX_DataType lambda, size_t trainingCount );

protected:

	virtual void calcOutput( const GX_DataVector & input, GX_DataVector * output ) const;

	virtual void backpropagate( const GX_DataVector & input, const GX_DataVector & output,
			const GX_DataVector & outDelta, GX_DataVector * inDelta ) const;

private:
	GX_DataMatrix mWeights;
	GX_DataVector mBiases;
};

