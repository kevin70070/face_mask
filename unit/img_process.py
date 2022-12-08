import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from skimage import io, draw
import numpy as np

def show(title,image):
    ii=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
    io.imshow(ii)
    print(f"{title} : ")
    io.show()


def mask_eye(im_base,img):       #捕捉眼睛
    eye_dict = 50
    eye_kernel = np.ones((3,15), np.uint8)
    temp = cv2.dilate(im_base, eye_kernel, iterations =2 )
    eye = img[0].copy()
    eye[temp==0]=[0,0,0]
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    (cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    aaa = np.zeros_like(eye)
    
    if len(cnts)==2:
        if cnts[0][0][0][0]>cnts[1][0][0][0]:
            bais = [eye_dict,-1*eye_dict]
        else:
            bais = [-1*eye_dict,eye_dict]
    else:
        bais = [0,0]
    cun = 0
    for c in cnts:
       
        
        clone = eye.copy()
        (x, y, w, h) = cv2.boundingRect(c)
        cx = x+w//2
        cy = y+h//2
       
        clone = clone[y:y+h,x:x+w]

        clone=cv2.resize(clone,dsize=None,fx=2,fy=2)
        #print('clone : ',clone.shape)
        
        new_h,new_w=clone.shape[0:2]
        new_x = cx-new_w//2+bais[cun]
      
        if bais[cun]<0:
            
            new_x = max(new_x,0)
           
        elif bais[cun]>0:
           
            new_x = min(new_x,aaa.shape[1])
            
        new_y = cy-new_h//2
        
        try:
            aaa[new_y:new_y+new_h,new_x:new_x+new_w] = clone
        except:
            pass
        
        cun+=1
        if cun>=2 :break
    
    return aaa

def mask_face(im_base,img):    #捕捉全臉
    kernel = np.ones((5,5), np.uint8)
    temp = cv2.dilate(im_base, kernel, iterations =2 )
    face = img[0].copy()
    face[temp!=1]=[0,0,0]
    
    return face

def mask_mount(im_base,img):   #捕捉嘴巴
    mount_kernel = np.ones((5,5), np.uint8)
    temp = cv2.dilate(im_base, mount_kernel, iterations =1 )
    mount = img[0].copy()
    mount[temp==0]=[0,0,0]
    gray = cv2.cvtColor(mount, cv2.COLOR_BGR2GRAY)

    (cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    for c in cnts:
        clone = mount.copy()
        (x, y, w, h) = cv2.boundingRect(c)
        cx = x+w//2
        cy = y+h//2
        clone = clone[y:y+h,x:x+w]
        clone=cv2.resize(clone,dsize=None,fx=1.2,fy=1.2)
        new_h,new_w=clone.shape[0:2]
        new_x = cx-new_w//2
        new_y = cy-new_h//2
        mount[new_y:new_y+new_h,new_x:new_x+new_w] = clone
    
    return mount


def get_maskimg(model,crop_img,image_size,debug=False):
    ori_w,ori_h,_=crop_img.shape
    #print('ddd : ',ori_h,ori_w)
    img = cv2.resize(crop_img,image_size)
    img = np.reshape(img,(1,)+image_size+(3,))
    ans=model.predict(img/255.0)[0]
    im_base = np.zeros((4,image_size[0],image_size[1]))

    for j in range(1,4):
        x = ans[:,:,j]      
        if j==2: im_base[j-1,x>0.05]=1
        elif j>2 : im_base[j-1,x>0.4]=1
        im_base[3,x>0.5]=1
        

    eye = mask_eye(im_base[1],img)
    if debug: show('eye',eye)
        
    mount = mask_mount(im_base[2],img)
    if debug: show('mount',mount)
        
    face = mask_face(im_base[3],img)
    if debug: show('face',face)

    eye = cv2.resize(eye,(ori_h,ori_w))
    mount = cv2.resize(mount,(ori_h,ori_w))
    face = cv2.resize(face,(ori_h,ori_w))
    
#     add_img = cv2.add(eye,mount)
#     gray_add_img = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY)  
    
#     mask = (add_img!=[0,0,0])[:,:,0]
#     face[mask] = add_img[mask]
#     #plt.imshow(face)
#     return face

    return {'eye':eye,'mount':mount,'face':face}

