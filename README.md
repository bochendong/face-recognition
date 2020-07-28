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
<p align="center">
	<img src="https://github.com/bochendong/face-recognition/raw/master/image/figure2.JPG"
        width="1500" height="600">
	<p align="center">
</p>
