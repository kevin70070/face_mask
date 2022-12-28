# face_mask
## 人臉提取
本專案為一個提取人臉的的方法，現階段已deeplabv3 plus 作為主要提取架構，下圖為deeplabv3 plus 架構圖
![image](https://user-images.githubusercontent.com/29139225/209787668-3075ef70-058a-47e6-9755-bb308f88a71d.png)

目前backbone 採用 resnet、res2nt、res2net with SE block ，你可以在models 中找到我們已經訓練好的模型

## 資料集
原始資料集包含了30000張 大小為 512 * 512 的人臉及標記圖片，其中有9個類別包含了如皮膚、鼻子、眼睛、眉毛、耳朵、嘴巴、嘴唇、頭髮、帽子、眼鏡、耳環、項鍊、頸部和布。  
你可以透過此連結找到原始CelebAMask_HQ 資料集 : http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html  

你也可以下載我們已經處理好的資料集，我們只用了其中的4類 : other,皮膚,眼睛,嘴巴 : 

## 環境
python==3.9  
tensorflow-gpu==2.8  

## 人臉體取效果
你可以在example.ipynb 中看到一個簡單的使用方法，使用我們已經訓練好的模型
![image](https://user-images.githubusercontent.com/29139225/209790258-af85c01f-1e35-41c6-a994-ff49bf5881fc.png)
