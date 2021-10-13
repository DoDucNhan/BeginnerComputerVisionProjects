import cv2
import numpy as np
import math
import pandas as pd
import argparse
import os
import sys
sys.setrecursionlimit(10**6)

# def get_args():
#     parser = argparse.ArgumentParser(description='Mosaic Image Generator')
#     parser.add_argument('--pixel_batch_size', type=int, default=1, required=True, help='control the detail of picture, lower means more detail but takes longer time to produce.')
#     parser.add_argument('--rmse_threshold', type=float, default=0.5, required=True, help='control the color similarity, try as lower as possible in the beginning. If adjust_threshold is 0 and if there is an error indicating "too lower threshold" then try to add the value slowly')
#     parser.add_argument('--target_PATH', type=str, required=True, help='PATH to the target image')
#     parser.add_argument('--source_PATH', type=str, required=True, help='PATH to the set of source images')
#     parser.add_argument('--output_PATH', type=str, required=True, help='PATH to the output image')
#     parser.add_argument('--allow_use_same_image', type=str, default='Y', choices = ['Y','N'], required=True, help='if true then the generator is allowed to use same picture many times')
#     parser.add_argument('--adjust_threshold', type=float, default=0.5, required=True, help='value of adjusted threshold for pixels which have rmse higher then the given initial threshold. If 0 then it will not adjusted')
#     return parser.parse_args()

def builder(source_PATH, input_image, output_PATH, pixel_batch_size=4, rmse_threshold=5,  
			allow_use_same_image=True, adjust_threshold=5):
	# args = get_args()

	# pixel_batch_size = args.pixel_batch_size
	# rmse_threshold = args.rmse_threshold
	# source_PATH = args.source_PATH
	# target_PATH = args.target_PATH
	# output_PATH = args.output_PATH
	# allow_use_same_image = True if args.allow_use_same_image =='Y' else False
	# adjust_threshold = args.adjust_threshold

	#Generate list of of relevant filenames per pixel batch size
	filenames, target_image_height, target_image_width, pixel_batch_size = find_filename_per_pixel_batch_size(input_image, pixel_batch_size,
		rmse_threshold, allow_use_same_image, adjust_threshold)

	#Adjust Mosaic Builder Size so it is the multiplies of pixel batch size
	size = check_mosaic_builder_size(size=20, pixel_batch_size=pixel_batch_size)
	print('\nUsed Mosaic Builder Size: {}\n'.format(size))


	#Constant Multiplier 
	k = int(size / pixel_batch_size)

	#Initiate Zeros Tensor for Mosaic
	img_concat = np.zeros((target_image_height*k, target_image_width*k, 3))

	#Create Mosaic Picture
	for i in range(0, target_image_height*k, size):
		for j in range(0, target_image_width*k, size):
			img = cv2.imread(os.path.join(source_PATH, filenames[0]), cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = np.array(cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA))

			if len(filenames) > 0:
				filenames.pop(0)
			img_concat[i:i+size, j:j+size,:] = img
			print('Finish Creating Mosaic for pixel %d, %d \r'%(i+size, j+size), end='')
	
	output = img_concat.astype(np.uint8)
	output = cv2.resize(output, (int(target_image_width*k/2), int(target_image_height*k/2)), cv2.INTER_AREA)
	cv2.imwrite(output_PATH, output)
	print('\nMosaic Image Saved! \n')
	return output


def find_filename_per_pixel_batch_size(input_image,  pixel_batch_size, threshold, allow_use_same_image, adjust_threshold):
	'''
	Function to return a list of 
	relevant filename per pixel batch size
	'''
    #Adjust Pixel Batch Size so it is the multiples of image's height and width
	pixel_batch_size = check_pixel_batch_size(pixel_batch_size, input_image)
	print('Used Pixel Batch Size: {}\n'.format(pixel_batch_size))

	#Import Database of Average RGB
	df = pd.read_csv('Avg_RGB_dataset.csv')

	height = input_image.shape[0]
	width = input_image.shape[1]
	filename_list = []

	print('Image Size: %dx%d \n'%(height, width))

	#looping for each image's pixel
	for i in range(0, height, pixel_batch_size):
	    for j in range(0, width, pixel_batch_size):
	        df, name = check_rmse(
				df,
	        	input_image[i:i+pixel_batch_size, j:j+pixel_batch_size, :],
	        	threshold=threshold,
	        	allow_repeated_use=allow_use_same_image,
	        	adjust_threshold=adjust_threshold)
	        filename_list.append(name)
	        print('Finish Creating Filename DataFrame for pixel %d, %d \r'%(i+pixel_batch_size, j+pixel_batch_size), end='')

	print('')
	return filename_list, height, width, pixel_batch_size


def check_pixel_batch_size(pixel_batch_size, img):
    '''
    Function to adjust Pixel Batch Size so it is the 
    multiples of image's height and width
    '''
    if (img.shape[0]%pixel_batch_size != 0) or (img.shape[1]%pixel_batch_size != 0):
        pixel_batch_size = math.gcd(img.shape[0], img.shape[1])
        
    print(pixel_batch_size)
    return pixel_batch_size


def check_rmse(df, batch_pixel, threshold, allow_repeated_use=False, adjust_threshold=1):
	'''
	Function to calculate rmse between each pixel and average RGB of images in database
	Input: 
	df: Database of Average RGB
	Pixel: pixel list of RGB
	threshold: threshold for RMSE
	'''
	while True:
		if df.empty:
			raise Exception("\n------Out of images, you should change \"allow_repeated_use\" to True-----\n")

		# Extract the average RGB from batch pixel
		pixel = [np.mean(batch_pixel[:, :, 0]), np.mean(batch_pixel[:, :, 1]), np.mean(batch_pixel[:, :, 2])]

		#Slice database with RGB value around the threshold
		slice_df = df[(df.avg_r <= pixel[0]+threshold) & (df.avg_r >= pixel[0]-threshold) & 
					(df.avg_g <= pixel[1]+threshold) & (df.avg_g >= pixel[1]-threshold) & 
					(df.avg_b <= pixel[2]+threshold) & (df.avg_b >= pixel[2]-threshold)][['avg_r', 'avg_g', 'avg_b']]
		it = slice_df.index.tolist()

		#Looping through the sliced database
		if len(slice_df) > 0:
			for i in it:
				rmse = np.sqrt(np.mean((slice_df.loc[i,['avg_r', 'avg_g', 'avg_b']] - pixel)**2)/3) 
				print('rmse: %f '%(rmse), end='')
				if rmse <= threshold:
					filename = df.loc[i, 'filename']
					if not allow_repeated_use:
						df = df.drop(i).reset_index(drop=True)
					break
			return df, filename
		else:
			if adjust_threshold > 0:
				threshold += adjust_threshold
				# return check_rmse(df, batch_pixel, threshold, allow_repeated_use)
			else:
				raise Exception('\n ----------------THRESHOLD TOO LOW---------------- \n')	
	


def check_mosaic_builder_size(size, pixel_batch_size):
	'''
	Function to adjust Mosaic Builder Size so it is the 
	multiplies of pixel batch size
	'''
	if (size%pixel_batch_size != 0):
		size = pixel_batch_size * int(size/pixel_batch_size)

	return size



if __name__ == '__main__':
	source, target, output = "test", "thumbs_up_down.jpg", "save.jpg"
	img = cv2.imread(target)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	builder(source, img, output)