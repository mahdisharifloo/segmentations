import pandas as pd 
import numpy as np 

train = open('dataset/train.txt').readlines()
train_box =  open('dataset/train_bbox.txt').readlines()
train_cate =  open('dataset/train_cate.txt').readlines()

test =  open('dataset/test.txt').readlines()
test_box =  open('dataset/test_bbox.txt').readlines()
test_cate =  open('dataset/test_cate.txt').readlines()

val = open('dataset/val.txt').readlines()
val_box =  open('dataset/val_bbox.txt').readlines()
val_cate =  open('dataset/val_cate.txt').readlines()


dict_train = {'im_path':train,'box':train_box,'cate':train_cate}
df_train = pd.DataFrame(dict_train)

dict_test = {'im_path':test,'box':test_box,'cate':test_cate}
df_test =  pd.DataFrame(dict_test)

dict_val = {'im_path':val,'box':val_box,'cate':val_cate}
df_val =  pd.DataFrame(dict_val)

df_train.to_csv('dataset/df_train.csv')
df_test.to_csv('dataset/df_test.csv')
df_val.to_csv('dataset/df_val.csv')