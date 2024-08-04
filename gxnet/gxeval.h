#pragma once

#include "gxcomm.h"
#include "gxnet.h"

class GX_Network;

void gx_eval( const char * tag, GX_Network & network, GX_DataMatrix & input, GX_DataMatrix & target, bool isDebug );

