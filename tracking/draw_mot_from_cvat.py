import cv2
import os
import random 

# 디렉토리 및 파일 경로 설정
dataset_dir = './datasets/C00_287_0001_clip1_mot'
img_dir = os.path.join(dataset_dir, 'img1')
gt_file = os.path.join(dataset_dir, 'gt/gt.txt')
labels_file = os.path.join(dataset_dir, 'gt/labels.txt')

# 클래스 라벨 로드
with open(labels_file, 'r') as f:
    labels = f.read().strip().split('\n')

# gt.txt 파일 파싱
annotations = {}
track_colors = {}
with open(gt_file, 'r') as f:
    for line in f:
        frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility = map(float, line.strip().split(','))
        frame_id, track_id, class_id = int(frame_id), int(track_id), int(class_id)
        
        if frame_id not in annotations:
            annotations[frame_id] = []
        
        annotations[frame_id].append((track_id, int(x), int(y), int(w), int(h), class_id))

        if track_id not in track_colors:
            track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# 이미지 파일 리스트 가져오기
image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.PNG')])

# 첫 번째 이미지로 프레임 크기 얻기
first_image_path = os.path.join(img_dir, image_files[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# 비디오 저장 설정
output_video_path = os.path.join(dataset_dir, 'output.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# 이미지 프레임에 바운딩 박스 그리기 및 비디오 작성
for img_file in image_files:
    frame_id = int(img_file.split('_')[1].split('.')[0])
    img_path = os.path.join(img_dir, img_file)
    frame = cv2.imread(img_path)
    
    if frame_id in annotations:
        for track_id, x, y, w, h, class_id in annotations[frame_id]:
            label = labels[class_id - 1]
            color = track_colors[track_id]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'ID: {track_id} {label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    video.write(frame)

# 비디오 객체 해제
video.release()
cv2.destroyAllWindows()
