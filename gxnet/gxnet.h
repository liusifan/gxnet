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

	bool calcOutput( const GX_DataVector input, bool isDebug,
			bool ignoreBias, GX_DataType * netOutput, GX_DataType * output );

private:
	GX_DataType mBias;

	GX_DataVector mWeights;
};

class GX_Layer {
public:
	GX_Layer( const int neuronCount, const int weightCount );
	~GX_Layer();

	GX_NeuronPtrVector & getNeurons();

private:
	GX_NeuronPtrVector mNeurons;
};

class GX_Network {
public:
	GX_Network( bool isDebug = false,
			bool ignoreBias = false,
			bool isSumMiniBatchGrad = false );
	~GX_Network();

	bool addLayer( const int neuronCount, const int weightCount );

	bool addLayer( const GX_DataMatrix & weights, const GX_DataVector & bias );

	GX_LayerPtrVector & getLayers();

	const GX_LayerPtrVector & getLayers() const;

	bool forward( const GX_DataVector & input,
			GX_DataMatrix * output );

	bool backward( const GX_DataVector & target,
			const GX_DataMatrix & output, GX_DataMatrix * delta );

	bool train( const GX_DataMatrix & input, const GX_DataMatrix & target,
			int epochCount, GX_DataType learningRate, GX_DataType lambda = 0 );

	bool train( const GX_DataMatrix & input, const GX_DataMatrix & target,
			bool isShuffle, int epochCount, int miniBatchCount,
			GX_DataType learningRate, GX_DataType lambda = 0 );

	void print();

private:

	bool collect( const GX_DataVector & input, const GX_DataMatrix & output,
			const GX_DataMatrix & delta, GX_DataMatrix * grad ) const;

	bool apply( const GX_DataMatrix & delta, const GX_DataMatrix & grad,
			int miniBatchCount, GX_DataType learningRate,
			GX_DataType lambda, int trainingCount );

private:
	GX_LayerPtrVector mLayers;
	bool mIsDebug;
	bool mIgnoreBias;
	bool mIsSumMiniBatchGrad; // use for debug
};

