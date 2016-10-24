# Joint Stereo and Flow
---

### Example
``
	python train.py --continue  0  --lr 1e-4 --model enet --gpus 0,1 --thread 30
``

### Option:

* Most options defined in config.py

### Prediction：

*prediction code is not included here.

###Data  organization

 - Class **Dataset** : if you want to add new dataset,please refer to **Dataset** and provide the following functions
 	- *init* : provide the directory of the dataset.
 	- *shapes* : the output of Dispnet is sensitive to the original shapes,so you need to  provide them.
 	- *get_data* : how to get the data
 
 - **Iterators**: We provide two kinds of data iterator,they are independent of specific dataset
 	- Class **Dataiter** :self-defined iterator based on python 
 	- Class **multi_imageRecord** : a wrapper of multi image records,If you want to tranfer stereo data to .rec file,a tool in util.py will help you.
 	- Iterators provide augmentation and some preprocessing like padding.
 	- If is_train is false,it will not augment data ,and if label exists it will provide original labels
 
### tools
we provide the following functions in util.py：

 - weighted median filter 
 - flow2color : plot optical flow
 - readPFM : read .pfm file
 - plot_velocity_vector : plot optical flow
 - get_imageRecord: transfer the dataset to .rec file





























    




