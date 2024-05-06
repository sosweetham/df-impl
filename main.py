# from deepface import DeepFace
# DeepFace.stream("database")

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    face_objs = DeepFace.extract_faces(frame, enforce_detection=False)
    for face_obj in face_objs:
        face_area = face_obj['facial_area']
        if (face_area['x'] != 0):
            cv2.rectangle(frame, (face_area['x'],face_area['y']), (face_area['x']+face_area['w'],face_area['y']+face_area['h']), (0,0,255), thickness=3, lineType=cv2.LINE_8)
            dfs = DeepFace.find(frame, './database/', model_name='VGG-Face', enforce_detection=False, distance_metric="euclidean_l2")
            for df in dfs:
                if len(df.index) > 0:
                    face_area = {
                        'x': df['source_x'].loc[df.index[0]],
                        'y': df['source_y'].loc[df.index[0]],
                        'w': df['source_w'].loc[df.index[0]],
                        'h': df['source_h'].loc[df.index[0]],
                        'identity': df['identity'].loc[df.index[0]]
                    }
                    face_crop = frame[face_area['y']:face_area['y']+face_area['h'], face_area['x']:face_area['x']+face_area['w']]
                    name = face_area['identity'].split('/')[-1].split('\\')[-1].split('_')[0]
                    try:
                        objs = DeepFace.analyze(face_crop, ['age', 'gender', 'race', 'emotion'])
                        details_text = ""
                        for obj in objs:
                            cv2.putText(face_crop, f'age: {obj['age']}', (0,70), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0))
                            cv2.putText(face_crop, f'gender: {obj['dominant_gender']}', (0,80), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0))
                            cv2.putText(face_crop, f'race: {obj['dominant_race']}', (0,90), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0))
                            cv2.putText(face_crop, f'emotion: {obj['dominant_emotion']}', (0,100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0))
                    except Exception as e:
                        continue
                    cv2.imshow(name, face_crop)
                    cv2.rectangle(frame, (face_area['x'],face_area['y']), (face_area['x']+face_area['w'],face_area['y']+face_area['h']), (0,255,0), thickness=3, lineType=cv2.LINE_8)
                    cv2.putText(frame, name, (face_area['x'],face_area['y']+face_area['h']), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0,255,0))
    cv2.imshow("stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()