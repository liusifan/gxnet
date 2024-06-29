from __future__ import print_function

import struct
import numpy as np
import random
import ctypes
import sys

from PIL import Image

def print_mnist( data ):
	tmp = data.reshape( -1 )
	for i in range( 28 ):
		for j in range( 28 ):
			if tmp[ i * 28 + j ] == 0 : print( "0", end="" )
			else: print( "1", end="" )
			#print( tmp[ i * 28 + j ], " ", end="" )
		print( "" )

def read_mnist_images( path ):

	fp = open( path, "rb" )
	buff = fp.read()
	fp.close()

	offset = 0

	fmt_header = '>iiii'
	magic_number, num_images, num_rows, num_cols = struct.unpack_from( fmt_header, buff, offset )

	print( "load {}, magic {}, count {}".format( path, magic_number, num_images ) )

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

	print( "load {}, magic {}, count {}".format( path, magic_number, label_num ) )

	offset += struct.calcsize(fmt_header)
	labels = []

	fmt_label = '>B'

	for i in range( label_num ):
		labels.append( struct.unpack_from( fmt_label, buff, offset )[ 0 ] )
		offset += struct.calcsize( fmt_label )
	return labels

def convert( image_path, label_path, suffix, transform ):

	images = read_mnist_images( image_path )
	labels = read_mnist_labels( label_path )

	num_images = len( images )

	rows, cols = 28, 28

	magic_images = 2051
	fmt_image = '>' + str( rows * cols ) + 'B'

	magic_labels = 2049

	fp = open( image_path + "." + suffix, "wb" )
	fp2 = open( label_path + "." + suffix, "wb" )

	buff = struct.pack( '>IIII', magic_images, num_images, rows, cols )
	fp.write( buff )

	buff = struct.pack( '>II', magic_labels, num_images )
	fp2.write( buff )

	for array, label in zip( images[ 0 : num_images ], labels[ 0 : num_images ] ):

		new_array = transform( array, label )

		new_buff = new_array.astype( np.uint8 ).ravel()

		buff = struct.pack( fmt_image, *new_buff )
		fp.write( buff )

		buff = struct.pack( '>B', label )
		fp2.write( buff )

	print( "save images {}".format( fp.name ) )
	print( "save labels {}".format( fp2.name ) )

	fp.close()
	fp2.close()

def rotate_mnist( rotation_range, image_path, label_path ):

	def transform( org_array, label ):
		while True:
			theta = float( random.uniform( rotation_range[ 0 ], rotation_range[ 1 ] ) )
			if 0 != theta: break

		org_img = Image.fromarray( np.uint8( org_array ) )
		new_img = org_img.rotate( theta, resample = Image.BICUBIC )

		return np.array( new_img )

	print( "rotation_range {}".format( rotation_range ) )

	convert( image_path, label_path, "rot", transform )

def recover_emnist( image_path, label_path ):

	def transform( org_array, label ):
		new_array = np.rot90( np.flip( org_array, axis = 0 ), -1 )

		# for debug
		if transform.count == 0:
			#print( label )
			#print_mnist( org_array )
			#print_mnist( new_array )
			transform.count += 1

		return new_array

	transform.count = 0

	convert( image_path, label_path, "rcv", transform )

if __name__ == '__main__':

	if( len( sys.argv ) < 4 ):
		print( "Usage: %s <rotate|recover> <images idx3 file> <labels idx1 file>\n" % ( sys.argv[ 0 ] ) )
		sys.exit( -1 )

	if sys.argv[ 1 ] == "rotate":
		rotate_mnist( [ -15, 15 ], sys.argv[ 2 ], sys.argv[ 3 ] )
	elif sys.argv[ 1 ] == "recover":
		recover_emnist( sys.argv[ 2 ], sys.argv[ 3 ] )
	else:
		print( "unknown command", sys.argv[ 1 ] )

