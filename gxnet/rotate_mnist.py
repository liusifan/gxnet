from __future__ import print_function

import struct
import numpy as np
import random
import ctypes

from PIL import Image

def print_mnist( data ):
	for i in range( 28 ):
		for j in range( 28 ):
			if data[ i * 28 + j ] == 0 : print( "0", end="" )
			else: print( "1", end="" )
			#print( data[ i * 28 + j ], " ", end="" )
		print( "" )

def read_mnist_images( path ):

	fp = open( path, "rb" )
	buff = fp.read()
	fp.close()

	offset = 0

	fmt_header = '>iiii'
	magic_number, num_images, num_rows, num_cols = struct.unpack_from( fmt_header, buff, offset )

	print( "load {}, magic：{}，count：{}".format( path, magic_number, num_images ) )

	offset += struct.calcsize( fmt_header )
	fmt_image = '>' + str( num_rows * num_cols ) + 'B'

	images = np.empty( ( num_images, num_rows, num_cols ) )

	for i in range( num_images ):
		im = struct.unpack_from( fmt_image, buff, offset )
		images[ i ] = np.array( im ).reshape( ( num_rows, num_cols ) )
		offset += struct.calcsize( fmt_image )
	return images

def read_mnist_labels( path ):
	fp = open( path, "rb" )
	buff = fp.read()
	fp.close()

	offset = 0

	fmt_header = '>ii'
	magic_number, label_num = struct.unpack_from(fmt_header, buff, offset)

	print( "load {}, magic：{}，count：{}".format( path, magic_number, label_num ) )

	offset += struct.calcsize(fmt_header)
	labels = []

	fmt_label = '>B'

	for i in range( label_num ):
		labels.append( struct.unpack_from( fmt_label, buff, offset )[ 0 ] )
		offset += struct.calcsize( fmt_label )
	return labels

def rotate( rotation_range, image_path, label_path ):

	print( "rotation_range {}".format( rotation_range ) )

	images = read_mnist_images( image_path )
	labels = read_mnist_labels( label_path )

	num_images = len( images )

	rows, cols = 28, 28

	magic_images = 2051
	fmt_image = '>' + str( rows * cols ) + 'B'

	magic_labels = 2049

	fp = open( image_path + ".rot", "wb" )
	fp2 = open( label_path + ".rot", "wb" )

	buff = struct.pack( '>IIII', magic_images, num_images, rows, cols )
	fp.write( buff )

	buff = struct.pack( '>II', magic_labels, num_images )
	fp2.write( buff )

	for buff, label in zip( images[ 0 : num_images ], labels[ 0 : num_images ] ):
		org_img = Image.fromarray( np.uint8( buff ) )

		while True:
			theta = float( random.uniform( rotation_range[ 0 ], rotation_range[ 1 ] ) )
			if 0 != theta: break

		new_img = org_img.rotate( theta, resample = Image.BICUBIC )

		new_buff = np.array( new_img ).ravel()

		buff = struct.pack( fmt_image, *new_buff )
		fp.write( buff )

		buff = struct.pack( '>B', label )
		fp2.write( buff )

	print( "save images {}".format( fp.name ) )
	print( "save labels {}".format( fp2.name ) )

	fp.close()
	fp2.close()

if __name__ == '__main__':

	rotate( [ -15, 15 ], "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte" )

