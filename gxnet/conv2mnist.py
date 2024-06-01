from __future__ import print_function

from PIL import Image, ImageFilter
import sys

def conv2mnist( path ):

	limitLen = 20

	orgImg = Image.open( path ).convert( 'L' )
	width = float( orgImg.size[ 0 ] )
	height = float( orgImg.size[ 1 ] )

	newImg = Image.new( 'L', ( 28, 28 ), ( 255 ) )

	if width > height:
		newHeight = int( round( ( limitLen / width * height ), 0 ) )
		if newHeight == 0: newHeight = 1
		top = int( round( ( ( 28 - newHeight ) / 2), 0 ) )

		orgImg = orgImg.resize( ( limitLen, newHeight ), Image.BICUBIC ).filter( ImageFilter.SHARPEN )
		newImg.paste( orgImg, ( 4, top ) )
	else:
		newWidth = int( round( ( limitLen / height * width ), 0 ) )
		if newWidth == 0:  newWidth = 1
		left = int( round( ( ( 28 - newWidth ) / 2 ), 0 ) )

		orgImg = orgImg.resize( ( newWidth, limitLen ), Image.BICUBIC ).filter( ImageFilter.SHARPEN )
		newImg.paste( orgImg, ( left, 4 ) )


	"""
	for x in range( newImg.size[ 0 ] ):
		for y in range( newImg.size[ 1 ] ):
			#print( newImg.getpixel( ( x, y ) ) )
			if newImg.getpixel( ( x, y ) ) != 255:
				newImg.putpixel( ( x, y ), 0 )
	"""

	newImg.save( path + ".bmp" )

	numList = list( newImg.getdata() )
	numList = [ ( 255 - x ) * 1.0 / 255.0 for x in numList ]

	fp = open( path + ".mnist", "w" )
	fp.write( ",".join( map( str, numList ) ) )
	fp.close()

	return numList

if( len( sys.argv ) < 2 ):
	print( "Usage: %s <img file>\n" % ( sys.argv[ 0 ] ) )
	sys.exit( -1 )

data = conv2mnist( sys.argv[ 1 ] )

for i in range( 0, 28 ):
	for j in range( 0, 28 ):
		if data[ i * 28 + j ] == 0 : print( "0", end="" )
		else: print( "1", end="" )
	print( "" )


