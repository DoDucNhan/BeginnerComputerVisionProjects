import numpy as np
import os
import pandas as pd
import argparse
from PIL import Image, ImageEnhance

# def get_args():
#     parser = argparse.ArgumentParser(description='AVG RGB Dataset Builder')
#     parser.add_argument('--SOURCE_PATH', type=str, required=True, help='Path to source images folder')
#     return parser.parse_args()

def build_dataframe_average_rgb(source):
    # args = get_args()

    df = pd.DataFrame(columns=['filename','avg_r','avg_g','avg_b'])
    
    # source = args.SOURCE_PATH
    for _, _, files in os.walk(source):
        filenames = files
        break
    
    length = len(filenames)
    index = 0
    
    print('')
    for filename in filenames:
        try:
            img = Image.open(os.path.join(source, filename))
            img_array = np.array(img)
            #Get the average value of Red, Green, and Blue
            #Original Image
            df = df.append({'filename': filename, 'avg_r': np.mean(img_array[:,:,0]), 'avg_g': np.mean(img_array[:,:,1]), 
                            'avg_b': np.mean(img_array[:,:,2])}, ignore_index=True)

            #RGB -> BGR Image
            bgr_img_array = img_array[:, :, ::-1]
            df = df.append({'filename': 'bgr_' + filename, 'avg_r': np.mean(bgr_img_array[:,:,0]), 'avg_g': np.mean(bgr_img_array[:,:,1]), 
                            'avg_b': np.mean(bgr_img_array[:,:,2])}, ignore_index=True)
            bgr_img = Image.fromarray(bgr_img_array)
            bgr_img.save(os.path.join(source, 'bgr_' + filename))

            # Enhanced Image
            img_enh = ImageEnhance.Contrast(img)
            img_enh = img_enh.enhance(1.8)
            img_enh_array = np.array(img_enh)
            df = df.append({'filename': 'enh_' + filename, 'avg_r': np.mean(img_enh_array[:,:,0]), 'avg_g': np.mean(img_enh_array[:,:,1]),
                            'avg_b': np.mean(img_enh_array[:,:,2])}, ignore_index=True)
            img_enh.save(os.path.join(source, 'enh_' + filename))

            # Grayscale Image
            grey_img = img.convert('L')
            grey_img_array = np.array(grey_img)
            df = df.append({'filename': 'gray_' + filename, 'avg_r': np.mean(grey_img_array), 'avg_g': np.mean(grey_img_array),
                            'avg_b': np.mean(grey_img_array)}, ignore_index=True)
            grey_img.save(os.path.join(source, 'gray_' + filename))

            index += 1
            print(('%.4f percents done \r') % (index*100/length), end='')
        except:
            index += 1
            print('\n Image {} Error'.format(filename))

    print('')
    df.to_csv('Avg_RGB_dataset.csv',index=False)

if __name__=='__main__':
    path = "test"
    build_dataframe_average_rgb(path)