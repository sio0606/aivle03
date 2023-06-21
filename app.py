# 라이브러리 불러오기 
import streamlit as st
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
from scipy.spatial import distance
from imutils import face_utils
import argparse
import imutils
import dlib
import face_recognition


# -------------------------------- setting  -------------------------------- #
# -------------------------------- 동일 인물 검사용 setting  -------------------------------#


cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # 얼굴 정면 인식 사전 학습 모델 


known_face_encodings = []
known_face_names = []

def draw_label(input_image, coordinates, label): # 사진에 얼굴 라벨링
    image = input_image.copy()
    (top, right, bottom, left) = coordinates
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 5)
    cv2.putText(image, label, (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    return image


def add_known_face(face_image_path, name): 
    face_image = cv2.imread(face_image_path)
    face_location = face_recognition.face_locations(face_image)[0]
    face_encoding = face_recognition.face_encodings(face_image)[0]
    
    detected_face_image = draw_label(face_image, face_location, name)
    
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    
    
    rgbFace = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)   
    # st.image(rgbFace)


def name_labeling(input_image):
    image = input_image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != "Unknown":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(image, (left, top), (right, bottom), color, 1)
        cv2.rectangle(image, (left, bottom - 10), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 3, bottom - 3), font, 0.2, (0, 0, 0), 1)
        

    if name != "Unknown":
        st.success('인증 성공')
    else:
        st.error('다시 시도해주세요.')
    
    
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    # st.image(rgbImage)
    
# -------------------------------- 눈깜박 + 캡쳐 setting  -------------------------------#


class Detector(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None
        self.original_frame = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.blink_count = 0
        self.blink_threshold = 3

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def transform(self, frame):
        img = frame.to_ndarray(format='bgr24')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.cascade.detectMultiScale(gray, 1.1, 3)
        self.original_frame = img.copy()
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        rects = self.detector(gray, 0)
        
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[self.lStart:self.lEnd]
            right_eye = shape[self.rStart:self.rEnd]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if self.blink_count > self.blink_threshold:
                cv2.putText(img, "Success", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif ear < 0.18:
                self.blink_count += 1
                cv2.putText(img, "Blink", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Not Blink", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.last_frame = img
        
        return img
    

# -------------------------------- 쏘카 첫 화면 설정  -------------------------------#

st.title('공유차량 운전면허 도용 방지 솔루션')
st.write('') # 공백
st.write('') # 공백
st.write('') # 공백


col1, col2, col3 = st.columns([0.2, 0.32, 0.2])

with col1:
    st.write('') # 가운데 정렬 
    
with col2:
       
    st.image('first.png') # 쏘카 최초 회면 이미지 
    
    st.expander("사용자 인증")

    with st.expander("사용자 인증하기"):
    
        st.info('눈을 깜박여주세요.')
        st.info('"Success" 문구가 나오면 얼굴을 촬영해주세요.')  
        st.info('화면 중앙에 얼굴을 위치시키고 밝은 환경에서 촬영해주세요.')
        
        webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=Detector, media_stream_constraints={"video": True, "audio": False})
        
        
        capture_button = st.button('인증하기')
        if capture_button and webrtc_ctx.video_transformer:
            captured_image = webrtc_ctx.video_transformer.original_frame
        
            
            if captured_image is not None:
                # 파일 이름, 경로 지정
                save_path = "captured_image.png" # 수정된 부분
                # 저장
                cv2.imwrite(save_path, captured_image)
                captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)  # 추가된 부분

            else:
                st.warning("캡처할 이미지가 없습니다. 웹캠이 작동되고 있습니까?")
        
            test_image_path = 'captured_image.png' # 캡처한 사진
            test_image = cv2.imread(test_image_path)
            rgbImage = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
            add_known_face("license.png", "face")
            name_labeling(test_image)


with col3:
    st.write('') # 가운데 정렬 


    

 
        
    
















