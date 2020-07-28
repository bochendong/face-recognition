# face-recognition
Face recognition using OpenCV and tensorflow

## Data collection
Call opencv's API function to realize face recognition
```Python
# Cat face recognition
classfier = cv2.CascadeClassifier("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalcatface.xml")
# Human face recognition
classfier = cv2.CascadeClassifier("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
```
Whenever a human face or cat face is detected in each frame, a rectangle centered on the recognized face is intercepted. Then save the captured image to the specified path
```Python
while cap.isOpened():
  ok, frame = cap.read()
  if not ok:            
    break       

  grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)           
        
  faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
  if len(faceRects) > 0:                                 
      for faceRect in faceRects:
          x, y, w, h = faceRect                        
          img_name = '%s/%d.jpg'%(path_name, num)                
          image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
          cv2.imwrite(img_name, image)                                                    
          num += 1                
          if num > (catch_pic_num):
              break

          cv2.rectangle(frame, (x - 10, y - 15), (x + w + 10, y + h + 15), color, 2)
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
      if num > (catch_pic_num): break                
                       
      cv2.imshow(window_name, frame)        
      c = cv2.waitKey(10)
      if c & 0xFF == ord('q'):
          break   
```
Examples:
<p align="center">
	<img src="https://github.com/bochendong/face-recognition/raw/master/image/figure1.png"
        width="1500" height="600">
	<p align="center">
</p>
<p align="center">
	<img src="https://github.com/bochendong/face-recognition/raw/master/image/figure3.JPG"
        width="1500" height="600">
	<p align="center">
</p>

## Load data
The main function of this step is to read the picture from the specified path and resize it to 64 * 64

### Load data from the specified path

```Python
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)                
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)  
                images.append(image)                
                labels.append(path_name)                                                
    return images,labels
    
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    images = np.array(images)  
    labels = np.array([0 if label.endswith('Bochen_Dong') else 1 for label in labels])    
    return images, labels
```

### Resize image

```Python
IMAGE_SIZE = 64
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)    
    
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left

    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])
    return cv2.resize(constant, (height, width))
```

## Train Model
The model is defined as:
<p align="center">
	<img src="https://github.com/bochendong/face-recognition/raw/master/image/model.png"
        width="600" height="1000">
	<p align="center">
</p>

The data argumentation is also implemented:
```Python
def train(self, dataset, batch_size = 20, nb_epoch = 10, data_augmentation = True):        
        sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)   
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy']) 
       
        if not data_augmentation:            
            self.model.fit(dataset.train_images, dataset.train_labels,
                           batch_size = batch_size, nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        else:            
            datagen = ImageDataGenerator(
                featurewise_center = False, samplewise_center  = False, featurewise_std_normalization = False, samplewise_std_normalization = False, zca_whitening = False, rotation_range = 20, width_shift_range  = 0.2, height_shift_range = 0.2, horizontal_flip = True, vertical_flip = False)         
            datagen.fit(dataset.train_images)                        
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))
```
## Predict

Example:
<p align="center">
	<img src="https://github.com/bochendong/face-recognition/raw/master/image/figure2.JPG"
        width="600" height="500">
	<p align="center">
</p>
