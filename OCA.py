from PIL import Image
import numpy as np

import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img 
    trg_img_np = trg_img 

    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

def apply_mask_to_enhanced_image(orig_img_path, mask_img_path, enhanced_img_path, output_img_path):
    orig_img = Image.open(orig_img_path).convert('RGB')  
    mask_img = Image.open(mask_img_path).convert('L') 
    enhanced_img = Image.open(enhanced_img_path).convert('RGB') 

    orig_img_array = np.array(orig_img)
    orig_img_array = orig_img_array[:,512:1024,:]
    mask_img_array = np.array(mask_img)
    enhanced_img_array = np.array(enhanced_img)
    
    orig_img_array = orig_img_array.transpose((2, 0, 1))
    enhanced_img_array = enhanced_img_array.transpose((2, 0, 1))
        
    # print("im_trg.shape : ", enhanced_img_array.shape ) # (3, 512, 1024)
    src_in_trg = FDA_source_to_target_np( orig_img_array, enhanced_img_array, L=0.01 )

    src_in_trg = src_in_trg.transpose((1,2,0))
    orig_img_array = np.clip(src_in_trg, 0, 255).astype(np.uint8)
    enhanced_img_array = enhanced_img_array.transpose((1,2,0))

    mask_13_14 = np.isin(mask_img_array, [11, 12, 13, 14, 15, 16, 17, 18])


    enhanced_img_array[mask_13_14] = orig_img_array[mask_13_14]
    output_img = Image.fromarray(enhanced_img_array)
    output_img.save(output_img_path)

    print(f"done: {output_img_path}")

import os
from pathlib import Path
from tqdm import tqdm
vis_folder = './OCA_demo_data/vis/'
labels_folder = './OCA_demo_data/labels/'
images_folder = './OCA_demo_data/images/'
output_folder = './OCA_demo_data/out_demo1/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for vis_file in tqdm(os.listdir(vis_folder)):
    if vis_file.endswith('.png'):
        base_name = vis_file.split('_genid')[0]  # exp：0a0eaeaf-9ad0c6dd
        orig_img_path = os.path.join(vis_folder, vis_file)
        mask_img_path = os.path.join(labels_folder, f"{base_name}_genid0_labelTrainIds.png")
        enhanced_img_path = os.path.join(images_folder, f"{base_name}_genid0.png")
        output_img_path = os.path.join(output_folder, f"{base_name}_fft_mix.png")
        apply_mask_to_enhanced_image(orig_img_path, mask_img_path, enhanced_img_path, output_img_path)

print("all done")

