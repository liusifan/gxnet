from __future__ import print_function

from PIL import Image, ImageFilter
import sys
import numpy as np

MNIST_SIZE = 28

def drop_margin( img ):
	begin_x, begin_y = img.size[ 0 ], img.size[ 1 ]
	end_x, end_y = 0, 0

	for x in range( img.size[ 0 ] ):
		for y in range( img.size[ 1 ] ):
			if img.getpixel( ( x, y ) ) != 255:
				begin_x, begin_y = min( begin_x, x ), min( begin_y, y )
				end_x, end_y = max( end_x, x ), max( end_y, y )

	end_x += 1
	end_y += 1

	img.crop( ( begin_x, begin_y, end_x, end_y ) )

def resize2mnist( org_img ):

	width = float( org_img.size[ 0 ] )
	height = float( org_img.size[ 1 ] )

	new_img = Image.new( 'L', ( MNIST_SIZE, MNIST_SIZE ), ( 255 ) )

	limit_len = 20
	margin_size = int( ( MNIST_SIZE - limit_len ) / 2 )

	if width > height:
		new_height = int( round( ( limit_len / width * height ), 0 ) )
		if new_height == 0: new_height = 1
		top = int( round( ( ( MNIST_SIZE - new_height ) / 2), 0 ) )

		org_img = org_img.resize( ( limit_len, new_height ), Image.BICUBIC ).filter( ImageFilter.SHARPEN )
		new_img.paste( org_img, ( margin_size, top ) )
	else:
		new_width = int( round( ( limit_len / height * width ), 0 ) )
		if new_width == 0:  new_width = 1
		left = int( round( ( ( MNIST_SIZE - new_width ) / 2 ), 0 ) )

		org_img = org_img.resize( ( new_width, limit_len ), Image.BICUBIC ).filter( ImageFilter.SHARPEN )
		new_img.paste( org_img, ( left, margin_size ) )

	num_array = np.array( new_img.getdata() )
	num_array = 255 - num_array
	num_array = num_array.reshape( ( MNIST_SIZE, MNIST_SIZE ) )

	new_img = Image.fromarray( np.uint8( num_array ) )

	return new_img

def conv2mnist( path ):

	org_img = Image.open( path ).convert( "RGBA" )
	pixel_data = org_img.load()

	# set all non-white pixels to black
	for x in range( org_img.size[ 0 ] ):
		for y in range( org_img.size[ 1 ] ):
			if any( pixel_data[ x, y ][ i ] < 220 for i in range( 4 ) ):
				pixel_data[ x, y ] = 0, 0, 0

	org_img = org_img.convert( 'L' )

	drop_margin( org_img )

	new_img = resize2mnist( org_img )

	# save the intermediate result for debug
	#new_img.save( path + ".bmp" )

	# save mnist data
	fp = open( path + ".mnist", "w" )
	fp.write( ",".join( map( str, list( new_img.getdata() ) ) ) )
	fp.close()

	return list( new_img.getdata() )

if __name__ == '__main__':

	if( len( sys.argv ) < 2 ):
		print( "Usage: %s <img file> <verbose>\n" % ( sys.argv[ 0 ] ) )
		sys.exit( -1 )

	data = conv2mnist( sys.argv[ 1 ] )

	if len( sys.argv ) > 2:
		for i in range( 0, MNIST_SIZE ):
			for j in range( 0, MNIST_SIZE ):
				if data[ i * MNIST_SIZE + j ] == 0 : print( "0", end="" )
				else: print( "1", end="" )
			print( "" )


