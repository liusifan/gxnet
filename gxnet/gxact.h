#pragma once

#include "gxcomm.h"

class GX_ActFunc {
public:
	enum { eNone, eSigmoid, eLeakyReLU, eTanh, eSoftmax };
public:
	GX_ActFunc( int type );

	~GX_ActFunc();

	int getType() const;

	void activate( const GX_DataVector & input, GX_DataVector * output ) const;

	void derivate( const GX_DataVector & output, GX_DataVector * outDelta ) const;

public:

	static GX_ActFunc * sigmoid();

	static GX_ActFunc * tanh();

	static GX_ActFunc * leakyReLU();

	static GX_ActFunc * softmax();

private:
	int mType;
};

