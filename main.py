from msilib.schema import Class
from turtle import update
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from djitellopy import tello
import time
from time import sleep
import cv2
import cvzone

from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture


me = tello.Tello()
me.connect()    
print(me.get_battery())
me.streamon()

class MainApp(App):
    def build(self):
        self.icon = "ss.png"

        self.img1=Image(allow_stretch = True, keep_ratio=False)
        self.video = BoxLayout(orientation='vertical',height='500dp',size_hint_y=None)
        self.video.add_widget(self.img1)
        Clock.schedule_interval(self.update, 1.0/33.0)

        main_layout = BoxLayout(orientation = "vertical")
        

        main_layout.add_widget(self.video)

        buttons = [
            ["Start", "Stop"],
            ["Up","Down"],
            ["Backward", "Forward"],
            ["Left", "Right"],
            ["Rotate C", "Rotate AnC"]
        ]


        for row in buttons:
            h_layout = BoxLayout(height='70dp')
            for label in row:
                button = Button(            
                    text = label, font_size=20, background_color="grey",
                    pos_hint={"center_x":0.5, "center_y": 0.5},
                )
                button.bind(on_press=self.on_button_press)
                h_layout.add_widget(button)

            main_layout.add_widget(h_layout)

        return main_layout

    def on_button_press(self, instance):
        button_text = instance.text

        self.speed = 50
        self.lr, self.fb, self.ud, self.yv = 0, 0, 0, 0
 
        if button_text == "Start": me.takeoff()
        if button_text == "Stop": me.land()

        if button_text == "Left": self.lr = -self.speed
        elif button_text == "Right": self.lr = self.speed

        if button_text == "Up": self.ud = self.speed
        elif button_text == "Down": self.ud = -self.speed

        
        if button_text == "Forward": self.fb = self.speed
        elif button_text == "Backward": self.fb = -self.speed

        if button_text == "RoRotate C": self.yv = self.speed
        elif button_text == "RoRotate AnC": self.yv = -self.speed

        self.vals = [self.lr, self.fb, self.ud, self.yv] 
        me.send_rc_control(self.vals[0], self.vals[1], self.vals[2], self.vals[3])


    def update(self, dt):
        self.img = me.get_frame_read().frame
        # cv2.imshow("Image", self.img)

        self.thres = 0.55
        self.nmsThres = 0.2
        # cap = cv2.VideoCapture(0)
        # cap.set(3, 640)
        # cap.set(4, 480)

        classNames = []
        classFile = 'coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().split('\n')
        print(classNames)
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = "frozen_inference_graph.pb"

        self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        
        self.classIds, self.confs, self.bbox = self.net.detect(self.img, confThreshold=self.thres, nmsThreshold=self.nmsThres)
        try:
            for classId, conf, box in zip(self.classIds.flatten(), self.confs.flatten(), self.bbox):
                cvzone.cornerRect(self.img, box)
                cv2.putText(self.img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)
        except:
            pass 

        # convert it to texture
        buf1 = cv2.flip(self.img, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(self.img.shape[1], self.img.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        
        


        # display image from the texture

        self.img1.texture = texture1

       

        cv2.imshow("Image", self.img)

    # while True:
    #     vals = on_button_press()
    #     me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    #     img = me.get_frame_read().frame
    #     img = cv2.resize(img,(360,240))
        
    #     cv2.imshow("image",img)
    #     cv2.waitKey(1)






if __name__ == '__main__':
    app = MainApp()
    app.run()