
  
import cv2
import os

path = '/home/pi/Desktop/piFace-master/FacialRecognition/dataset'
files = os.listdir(path)
print('Profile list:')
for f in files:
    print(f)
print('')
name = input("Write the name of your profile: ")
exist = False
while True:
    exist = False
    for f in files:
        if(f==name):
            print('this profile exists in the datebase')
            exist = True
    if(exist == False):
        break
    else:
        name = input('Please provide a different profile name: ')
        
path = '/home/pi/Desktop/piFace-master/FacialRecognition/dataset/'
path = os.path.join(path,name)
os.mkdir(path)
print('Profile created')


cam = cv2.VideoCapture(0)

cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 640, 480)

img_counter = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, -1) # Flip vertically
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("press space to take a photo", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "dataset/"+ name +"/" + name + "0.{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
