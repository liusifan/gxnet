#pragma once

#include <vector>

typedef double GX_DataType;
typedef std::vector< GX_DataType > GX_DataVector;
typedef std::vector< GX_DataVector > GX_DataMatrix;

class GX_Neuron;
class GX_Layer;

typedef std::vector< GX_Neuron * > GX_NeuronPtrVector;
typedef std::vector< GX_Layer * > GX_LayerPtrVector;

class GX_Neuron {
public:
	GX_Neuron( const int weightCount );
	~GX_Neuron();

	GX_DataVector & getWeights();

	const GX_DataType getBias();
	void setBias( GX_DataType bias );

	GX_DataType calcOutput( const GX_DataVector & input, bool isDebug, bool ignoreBias );

private:
	GX_DataType mBias;

	GX_DataVector mWeights;
};

class GX_Layer {
public:
	enum { eSigmoid, eReLU, eTanh, eSoftmax };

public:
	GX_Layer( const int neuronCount, const int weightCount, int actFuncType );
	~GX_Layer();

	GX_NeuronPtrVector & getNeurons();

	int getActFuncType() const;

	void calcOutput( const GX_DataVector & input, bool isDebug,
			bool ignoreBias, GX_DataVector * output ) const;

	void derivative( const GX_DataVector & output, const GX_DataVector & dOutput,
			GX_DataVector * delta ) const;

private:
	void activate( GX_DataVector * output ) const;

private:
	GX_NeuronPtrVector mNeurons;
	int mActFuncType;
};

class GX_Network {
public:
	enum { eMeanSquaredError, eCrossEntropy };

public:
	GX_Network( int costFuncType = eMeanSquaredError );

	~GX_Network();

	void setDebug( bool flag );

	void setDebugBackward( bool flag );

	bool addLayer( const int neuronCount, const int weightCount, int actFuncType = GX_Layer::eSigmoid );

	bool addLayer( const GX_DataMatrix & weights, const GX_DataVector & bias, int actFuncType = GX_Layer::eSigmoid );

	void setLossFuncType( int lossFuncType );

	int getLossFuncType() const;

	GX_LayerPtrVector & getLayers();

	const GX_LayerPtrVector & getLayers() const;

	bool forward( const GX_DataVector & input,
			GX_DataMatrix * output );

	bool backward( const GX_DataVector & target,
			const GX_DataMatrix & output, GX_DataMatrix * delta );

	bool train( const GX_DataMatrix & input, const GX_DataMatrix & target,
			int epochCount, GX_DataType learningRate, GX_DataType lambda = 0,
			GX_DataVector * losses = nullptr );

	bool train( const GX_DataMatrix & input, const GX_DataMatrix & target,
			bool isShuffle, int epochCount, int miniBatchCount,
			GX_DataType learningRate, GX_DataType lambda = 0, GX_DataVector * losses = nullptr );

	void print();

private:

	bool collect( const GX_DataVector & input, const GX_DataMatrix & output,
			const GX_DataMatrix & delta, GX_DataMatrix * grad ) const;

	bool apply( const GX_DataMatrix & delta, const GX_DataMatrix & grad,
			int miniBatchCount, GX_DataType learningRate,
			GX_DataType lambda, int trainingCount );

	bool forwardInternal( const GX_DataVector & input,
			GX_DataMatrix * output );

	void calcOutputDelta( const GX_Layer & layer, const GX_DataVector & target,
			const GX_DataVector & output, GX_DataVector * delta );

	GX_DataType calcLoss( const GX_DataVector & target, const GX_DataVector & output );

private:

	static void initGradMatrix( const GX_LayerPtrVector & layers,
			GX_DataMatrix * miniBatchGrad, GX_DataMatrix * grad );

	static void initOutputAndDeltaMatrix( const GX_LayerPtrVector & layers,
			GX_DataMatrix * output, GX_DataMatrix * miniBatchDelta, GX_DataMatrix * delta );

private:
	int mLossFuncType;
	GX_LayerPtrVector mLayers;
	bool mIsDebug;
	bool mIsDebugBackward;
};

