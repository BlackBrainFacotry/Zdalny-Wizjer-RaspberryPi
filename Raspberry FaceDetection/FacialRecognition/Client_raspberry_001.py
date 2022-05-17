import websockets
import pickle
import struct
import imutils
import cv2
import numpy as np
import os
import asyncio
from imutils import paths
from PIL import Image
from websockets import connect
import time


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0


#names = ['K','None'] 
names = list(paths.list_images('dataset'))



#server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#host_name  = socket.gethostname()
#host_ip = '192.168.0.109' # Enter the server IP
#print('HOST IP:',host_ip)
#port = 50000
#socket_address = (host_ip,port)
#server_socket.bind(socket_address)
#server_socket.listen()
#print("Listening at",socket_address)

#client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip='192.168.0.108'
port=5050
#client_socket.connect((host_ip,port))
uri="ws://192.168.0.108:5050/"
#client_socket = websockets.websocket()
#client_socket = websockets.connect(uri)

date = b""
payload_size = struct.calcsize("Q")

cam = cv2.VideoCapture(0)
minW = 0
minH = 0
async def start_video_stream(client_socket):
    #client_socket,addr = server_socket.accept()
    
    global cam
    global minW
    global minH
    try:
        print('START')
        if client_socket:
            try:
                if(cam.isOpened()):
                    ret,frame = cam.read()
                    #print('frame reading...')
                    frame = cv2.flip(frame, -1) # Flip vertically

                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    #print('facial detecting...')
                    faces = faceCascade.detectMultiScale( 
                        gray,
                        scaleFactor = 1.2,
                        minNeighbors = 5,
                        minSize = (int(minW), int(minH)),
                       )
                    
                    for(x,y,w,h) in faces:

                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                        print('Znaleziono twarz!')
                        print('========================================')
                        # Check if confidence is less them 100 ==> "0" is perfect match 
                        if (confidence < 100):
                            id = names[id]
                            confidence = "  {0}%".format(round(100 - confidence))
                            print('Rozpoznano twarz!')
                            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                        else:
                            id = "unknown"
                            confidence = "  {0}%".format(round(100 - confidence))
                        print('text input...')
                        cv2.putText(frame, str("Joana"), (x+5,y-5), font, 1, (255,255,255), 2)
                        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
                                
                    #frame  = imutils.resize(frame,width=320)
                   # a = pickle.dumps(frame)
                    
                    #message = struct.pack("Q",len(a))+a
                    picture_bytes = cv2.imencode('.jpg',frame)[1].tobytes()
                    command = "{:<100}".format("Get_picture")
                    message = command.encode() + picture_bytes
                    try:
                        #client_socket.sendall('message'.encode())
                        #xxx = 'message'.encode()
                        #await client_socket.send("222222222aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                        if(message != None):
                            if (len(message) >10):
                                await client_socket.send(message) #'message'.encode())
                            else:
                                print(len(message))
                        else:
                            print('message NULL')
                    except Exception as e:
                        print(e)
                        print('____________')
                        await client_socket.close()
                        #break
                #print('send message')
                    cv2.imshow("TRANSMITTING TO CACHE SERVER",frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key ==ord('q'):
                        client_socket.close()
                        #break
                #time.sleep(0.1)
            except Exception as e:
                print( e)
                
    except Exception as e:
            print(e)
async def train_model(client_socket):
	# Path for face image database
	path = 'dataset'
	if not os.path.exists('trainer'):os.makedirs('trainer')
		
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

	

	print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
	faces,ids = getImagesAndLabels(path)
	recognizer.train(faces, np.array(ids))

	# Save the model into trainer/trainer.yml
	recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

	# Print the numer of faces trained and end program
	print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

	message = "{:<100}".format("Get_train_model")
	await client_socket.send(message.encode())            
		
		
# function to get the images and label data
def getImagesAndLabels(path):
	
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
	#imagePaths = [os.path.join(path,f) for f in os.listdir(path) ]
	#imagePaths =  [os.path.join([os.path.join(path,f) for f in os.listdir(path) ],g ) for  g in os.listdir[os.getcwd[ for k in os.path.join(path,f) for f in os.listdir(path)] ] ] 
	#imagePaths = [os.path.join(path,f) for f in os.listdir[path] if os.path.isdir(os.path.join(path,f) ) ]
	#filter(lambda x: os.path.isdir(os.path.join(path,x)),os.listdir(path) )
	for i in os.listdir(path):
		j = os.path.join(path,i)
		imagePaths = [os.path.join(j,f) for f in os.listdir(j)]
	faceSamples=[]
	ids = []
	for imagePath in imagePaths:

		PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
		img_numpy = np.array(PIL_img,'uint8')

		id = int(os.path.split(imagePath)[-1].split(".")[1])
		faces = detector.detectMultiScale(img_numpy)

		for (x,y,w,h) in faces:
			faceSamples.append(img_numpy[y:y+h,x:x+w])
			ids.append(id)

	return faceSamples,ids 
	
	
async def ff(ws):
        #await ws.send("AAAAAAAAAAAAaaaaxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        await start_video_stream(ws)

async def return_name_client(client_socket):
	comand= "Get_name"
	name = "raspberry"
	message = "{:<100}{:<100}".format(comand,name)
	print(message)
	await client_socket.send(message.encode())

        
async def return_profile_list(client_socket):
    path = '/home/pi/Desktop/piFace-master/FacialRecognition/dataset'
    path_list = os.listdir(path)
    list_name = " "
    comand= "Get_person_list"
    print('Profile list:')
    for f in path_list:
        print(f)
        list_name = list_name + f
        list_name = list_name  + " " 
    print(list_name)
    message = "{:<100}{:<100}".format(comand,list_name)
    print(message)
    await client_socket.send(message.encode())
          

	
async def main():
	addr=host_ip
	global cam
	global minW
	global minH
	camera = True
	if camera == True:
		#cam = cv2.VideoCapture(0)
		cam.set(3, 480)# set video widht
		cam.set(4, 640)# set video height
		# Define min window size to be recognized as a face
		minW = 0.1*cam.get(3)
		minH = 0.1*cam.get(4)
		
	async with connect(uri,ping_interval=None) as websocket:
		await return_name_client(websocket)
		while True:
			###
			data = await websocket.recv()
			command = data[0:100]
			command = command.decode()
			command = command.strip()
			print(command)
			if(command == "Get_picture"):				
				await ff(websocket)
			if(command== "Get_person_list"):
				await return_profile_list(websocket)
			if(command == "Get_train_model"):
				print('@@@@@@@')
				await train_model(websocket)
			else:
				#print(data)
				print("command")

#asyncio.run(main
print('START')
asyncio.get_event_loop().run_until_complete(main())
print('asdfdasdfa')
asyncio.get_event_loop().run_forever()
