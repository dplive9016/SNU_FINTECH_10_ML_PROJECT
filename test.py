import os
import glob
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# 액션 카테고리 정의 및 영어 변환
ACTION_CATEGORIES = {
    # 1. 졸음/피로 및 무기력 관련
    "꾸벅꾸벅졸다": 0,
    "눈깜빡이기": 0,
    "눈비비기": 0, 
    "하품": 0,
    "힐끗거리다": 0,
    "고개를돌리다": 0,
    "고개를좌우로흔들다": 0,
    "몸못가누기": 0,
    "허리굽히다": 0,
    "옆으로기대다": 0,
    
    # 2. 운전자 방해 및 위험 행동
    "운전자를향해손을뻗다": 1,
    "운전자를향해발을뻗다": 1,
    "운전자의핸들조작방해하기": 1,
    "핸들을놓치다": 1,
    "핸들을흔들다": 1,
    "침뱉기": 1,
    "일어서다": 1,
    "차량의문열기": 1,
    "창문을열다": 1,
    
    # 3. 물건 사용/조작 관련
    "무언가를마시다": 2,
    "무언가를보다": 2,
    "무언가를쥐다": 2,
    "물건을쥐다/휘두르다": 2,
    "핸드폰귀에대기": 2,
    "핸드폰쥐기": 2,
    "중앙으로손을뻗다": 2,
    
    # 4. 신체 동작 및 접촉/상호작용
    "손을뻗다": 3,
    "어깨를두드리다": 3,
    "뺨을때리다": 3,
    "허벅지두드리기": 3,
    "팔주무르기": 3,
    "몸을돌리다": 3,
    "중앙을쳐다보기": 3,
    "운전하다": 3,
}

# 한국어 액션 -> 영어 변환 매핑
ACTION_TO_ENGLISH = {
    "꾸벅꾸벅졸다": "Dozing",
    "눈깜빡이기": "Blinking",
    "눈비비기": "Rubbing Eyes",
    "하품": "Yawning",
    "힐끗거리다": "Glancing",
    "고개를돌리다": "Turning Head",
    "고개를좌우로흔들다": "Shaking Head",
    "몸못가누기": "Unstable Body",
    "허리굽히다": "Bending",
    "옆으로기대다": "Leaning",
    
    "운전자를향해손을뻗다": "Reaching Driver",
    "운전자를향해발을뻗다": "Extending Leg",
    "운전자의핸들조작방해하기": "Interfering Steering",
    "핸들을놓치다": "Losing Wheel",
    "핸들을흔들다": "Shaking Wheel",
    "침뱉기": "Spitting",
    "일어서다": "Standing Up",
    "차량의문열기": "Opening Door",
    "창문을열다": "Opening Window",
    
    "무언가를마시다": "Drinking",
    "무언가를보다": "Looking at Something",
    "무언가를쥐다": "Holding Something",
    "물건을쥐다/휘두르다": "Waving Object",
    "핸드폰귀에대기": "Phone to Ear",
    "핸드폰쥐기": "Holding Phone",
    "중앙으로손을뻗다": "Reaching Center",
    
    "손을뻗다": "Extending Hand",
    "어깨를두드리다": "Tapping Shoulder",
    "뺨을때리다": "Slapping Cheek",
    "허벅지두드리기": "Tapping Thigh",
    "팔주무르기": "Rubbing Arm",
    "몸을돌리다": "Turning Body",
    "중앙을쳐다보기": "Looking at Center",
    "운전하다": "Driving"
}

# 카테고리 이름 정의 (영어)
CATEGORY_NAMES = {
    0: "Drowsiness/Fatigue",
    1: "Dangerous Behavior",
    2: "Using Objects",
    3: "Body Movement"
}

# 카테고리별 색상 정의 (B, G, R 형식)
CATEGORY_COLORS = {
    0: (255, 0, 0),    # 빨강 - 졸음/피로
    1: (0, 0, 255),    # 파랑 - 운전자 방해 및 위험
    2: (0, 255, 0),    # 초록 - 물건 사용
    3: (255, 255, 0),  # 청록 - 신체 동작
}

# 얼굴 바운딩 박스용 색상
FACE_COLOR = (255, 165, 0)  # 주황색
# YOLO 바운딩 박스용 색상
YOLO_COLOR = (0, 255, 255)  # 노란색

def load_yolo_bboxes(yolo_file, img_width, img_height):
    """
    YOLO 형식 파일에서 바운딩 박스 정보를 로드하고 픽셀 좌표로 변환합니다.
    
    반환:
        yolo_bboxes: 바운딩 박스 리스트 [(xmin, ymin, xmax, ymax, class_id), ...]
    """
    yolo_bboxes = []
    
    if not os.path.exists(yolo_file):
        return yolo_bboxes
    
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            yolo_bboxes.append((xmin, ymin, xmax, ymax, class_id))
    
    return yolo_bboxes

def draw_yolo_bboxes(img, yolo_bboxes, category_names=None):
    """
    YOLO 바운딩 박스를 이미지에 그립니다.
    """
    # 이미지 복사
    result_img = img.copy()
    
    for xmin, ymin, xmax, ymax, class_id in yolo_bboxes:
        # 바운딩 박스 그리기 (점선 효과)
        for i in range(0, int(xmax - xmin), 5):
            cv2.line(result_img, (int(xmin + i), int(ymin)), (int(min(xmin + i + 3, xmax)), int(ymin)), YOLO_COLOR, 2)
            cv2.line(result_img, (int(xmin + i), int(ymax)), (int(min(xmin + i + 3, xmax)), int(ymax)), YOLO_COLOR, 2)
        
        for i in range(0, int(ymax - ymin), 5):
            cv2.line(result_img, (int(xmin), int(ymin + i)), (int(xmin), int(min(ymin + i + 3, ymax))), YOLO_COLOR, 2)
            cv2.line(result_img, (int(xmax), int(ymin + i)), (int(xmax), int(min(ymin + i + 3, ymax))), YOLO_COLOR, 2)
        
        # 클래스 라벨 표시
        label = f"YOLO: Class {class_id}"
        if category_names and class_id in category_names:
            label = f"YOLO: {category_names[class_id]}"
        
        # 텍스트 배경
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            result_img,
            (int(xmin), int(ymax) + 5),
            (int(xmin) + text_size[0], int(ymax) + 25),
            (0, 0, 0),
            -1
        )
        
        # 텍스트 그리기
        cv2.putText(
            result_img,
            label,
            (int(xmin), int(ymax) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return result_img

def convert_json_to_yolo_format(json_file_path, output_dir=None, img_base_dir=None, visualize=True):
    """
    JSON 파일을 YOLO v4 형식으로 변환하고 바운딩 박스를 시각화합니다.
    신체와 얼굴 바운딩 박스를 모두 YOLO 형식으로 저장합니다.
    
    매개변수:
        json_file_path (str): JSON 파일 경로
        output_dir (str): 출력 디렉토리 (기본값: None, 자동 생성)
        img_base_dir (str): 이미지 기본 디렉토리 (기본값: None)
        visualize (bool): 바운딩 박스 시각화 여부 (기본값: True)
    """
    # JSON 파일 디렉토리와 파일명 추출
    json_dir = os.path.dirname(json_file_path)
    file_name = os.path.basename(json_file_path)
    scene_id = os.path.splitext(file_name)[0]
    
    # 출력 디렉토리가 None이면 JSON 파일이 있는 디렉토리에 yolo_labels 폴더 생성
    if output_dir is None:
        output_dir = os.path.join(json_dir, f"{scene_id}_yolo_labels")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 시각화 디렉토리 생성
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        compare_dir = os.path.join(output_dir, "comparisons")
        os.makedirs(compare_dir, exist_ok=True)
    
    # JSON 파일 읽기
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_file_path}: {e}")
        return
    
    # JSON 구조 검증
    if "scene" not in data or "data" not in data["scene"]:
        print(f"Invalid JSON structure in {json_file_path}: Missing 'scene' or 'data' field")
        return
    
    # 클래스 파일 생성
    with open(os.path.join(output_dir, "classes.txt"), 'w', encoding='utf-8') as f:
        for category_id in sorted(CATEGORY_NAMES.keys()):
            f.write(f"{CATEGORY_NAMES[category_id]}\n")
    
    # 이미지 데이터 처리
    visualization_count = 0
    for img_data in data["scene"]["data"]:
        if "img_name" not in img_data or "occupant" not in img_data:
            continue  # 필요한 필드가 없으면 건너뜀
            
        img_name = img_data["img_name"]
        label_file_name = os.path.splitext(img_name)[0] + ".txt"
        label_file_path = os.path.join(output_dir, label_file_name)
        
        # 시각화를 위한 이미지 로드
        vis_img = None
        img_width, img_height = 1280, 720  # 기본 이미지 크기 가정
        
        if visualize and img_base_dir:
            # 가능한 이미지 경로 패턴
            img_patterns = [
                os.path.join(img_base_dir, "**", img_name),
                os.path.join(img_base_dir, "**", "images", img_name),
                os.path.join(img_base_dir, img_name),
                os.path.join(img_base_dir, "*", img_name),
                os.path.join(img_base_dir, "**", "*", img_name)
            ]
            
            # 이미지 찾기
            img_path = None
            for pattern in img_patterns:
                matching_files = glob.glob(pattern, recursive=True)
                if matching_files:
                    img_path = matching_files[0]
                    break
            
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img_height, img_width = img.shape[:2]
                    vis_img = img.copy()
                    compare_img = img.copy()  # 비교용 이미지
                else:
                    print(f"Warning: Unable to read image: {img_path}")
            else:
                print(f"Warning: Image file not found: {img_name}")
        
        # YOLO 형식으로 저장할 바운딩 박스 목록
        yolo_annotations = []
        
        # JSON 바운딩 박스 정보 수집
        json_body_bboxes = []
        json_face_bboxes = []
        
        with open(label_file_path, 'w', encoding='utf-8') as f:
            for occupant in img_data["occupant"]:
                # 필수 필드 확인
                if "body_b_box" not in occupant or "action" not in occupant:
                    continue
                
                # 동작에 따른 카테고리 적용
                action = occupant["action"]
                
                # 카테고리 매핑 - 맵에 없는 액션은 기본값 3(신체 동작)으로 처리
                category_id = ACTION_CATEGORIES.get(action, 3)
                
                # 1. 신체 바운딩 박스 처리
                try:
                    bbox = occupant["body_b_box"]
                    
                    # 바운딩 박스 형식 처리
                    if len(bbox) == 4:
                        xmin, ymin, xmax, ymax = bbox
                        
                        # JSON 바운딩 박스 정보 저장
                        json_body_bboxes.append((xmin, ymin, xmax, ymax, action, category_id))
                        
                        # YOLO 형식으로 변환: [class_id] [x_center] [y_center] [width] [height]
                        x_center = (xmin + xmax) / 2 / img_width
                        y_center = (ymin + ymax) / 2 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height
                        
                        # YOLO 형식으로 저장 - 신체 바운딩 박스
                        yolo_annotation = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        f.write(f"{yolo_annotation}\n")
                        yolo_annotations.append(yolo_annotation)
                        
                        # 시각화 (이미지가 있는 경우만)
                        if vis_img is not None:
                            color = CATEGORY_COLORS.get(category_id, (200, 200, 200))  # 기본 색상은 회색
                            
                            # 바운딩 박스 그리기
                            cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                            
                            # 영어로 변환된 클래스 이름 표시
                            english_action = ACTION_TO_ENGLISH.get(action, "Unknown Action")
                            label = f"{english_action} (Body)"
                            
                            # 텍스트 배경 (가독성 향상)
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(
                                vis_img, 
                                (int(xmin), int(ymin) - 25),
                                (int(xmin) + text_size[0], int(ymin) - 5),
                                (0, 0, 0),  # 검은색 배경
                                -1  # 채우기
                            )
                            
                            # 텍스트 그리기
                            cv2.putText(
                                vis_img, 
                                label, 
                                (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255),  # 흰색 텍스트
                                2
                            )
                            
                            # 비교 이미지에도 JSON 바운딩 박스 그리기
                            cv2.rectangle(compare_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                            cv2.putText(
                                compare_img, 
                                f"JSON Body: {english_action}", 
                                (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                color, 
                                2
                            )
                    else:
                        print(f"Warning: Invalid body bounding box format: {bbox} in {img_name}")
                except Exception as e:
                    print(f"Error processing body bounding box in {img_name}: {e}")
                    continue
                
                # 2. 얼굴 바운딩 박스 처리 - 신체와 동일한 클래스 사용
                if "face_b_box" in occupant:
                    try:
                        face_bbox = occupant["face_b_box"]
                        
                        if len(face_bbox) == 4:
                            # 두 가지 형식 처리: [x, y, width, height] 또는 [x1, y1, x2, y2]
                            if len(face_bbox) == 4:
                                # face_b_box 형식 확인 ([x, y, width, height] 또는 [x1, y1, x2, y2])
                                if face_bbox[2] < img_width and face_bbox[3] < img_height:
                                    # [x, y, width, height] 형식으로 가정
                                    face_xmin, face_ymin, face_width, face_height = face_bbox
                                    face_xmax = face_xmin + face_width
                                    face_ymax = face_ymin + face_height
                                else:
                                    # [x1, y1, x2, y2] 형식으로 가정
                                    face_xmin, face_ymin, face_xmax, face_ymax = face_bbox
                                
                                # JSON 바운딩 박스 정보 저장
                                json_face_bboxes.append((face_xmin, face_ymin, face_xmax, face_ymax, action, category_id))
                                
                                # YOLO 형식으로 변환 - 얼굴 바운딩 박스
                                face_x_center = (face_xmin + face_xmax) / 2 / img_width
                                face_y_center = (face_ymin + face_ymax) / 2 / img_height
                                face_width = (face_xmax - face_xmin) / img_width
                                face_height = (face_ymax - face_ymin) / img_height
                                
                                # 얼굴도 신체와 같은 카테고리 ID 사용
                                face_category_id = category_id
                                
                                # YOLO 형식으로 저장 - 얼굴 바운딩 박스
                                yolo_annotation = f"{face_category_id} {face_x_center:.6f} {face_y_center:.6f} {face_width:.6f} {face_height:.6f}"
                                f.write(f"{yolo_annotation}\n")
                                yolo_annotations.append(yolo_annotation)
                                
                                # 시각화 (이미지가 있는 경우만)
                                if vis_img is not None:
                                    # 얼굴 바운딩 박스 그리기 (주황색으로 구분)
                                    cv2.rectangle(vis_img, (int(face_xmin), int(face_ymin)), 
                                                (int(face_xmax), int(face_ymax)), FACE_COLOR, 2)
                                    
                                    # 얼굴 라벨 표시
                                    face_label = f"{english_action} (Face)"
                                    
                                    # 얼굴 텍스트 배경
                                    face_text_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(
                                        vis_img, 
                                        (int(face_xmin), int(face_ymin) - 25),
                                        (int(face_xmin) + face_text_size[0], int(face_ymin) - 5),
                                        (0, 0, 0),  # 검은색 배경
                                        -1  # 채우기
                                    )
                                    
                                    # 얼굴 텍스트 그리기
                                    cv2.putText(
                                        vis_img, 
                                        face_label, 
                                        (int(face_xmin), int(face_ymin) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, 
                                        (255, 255, 255),  # 흰색 텍스트
                                        2
                                    )
                                    
                                    # 비교 이미지에도 JSON 얼굴 바운딩 박스 그리기
                                    cv2.rectangle(compare_img, (int(face_xmin), int(face_ymin)), 
                                                (int(face_xmax), int(face_ymax)), FACE_COLOR, 2)
                                    cv2.putText(
                                        compare_img, 
                                        f"JSON Face: {english_action}", 
                                        (int(face_xmin), int(face_ymin) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, 
                                        FACE_COLOR, 
                                        2
                                    )
                            else:
                                print(f"Warning: Invalid face bounding box format: {face_bbox} in {img_name}")
                    except Exception as e:
                        print(f"Error processing face bounding box in {img_name}: {e}")
                        continue
        
        # 시각화 이미지 저장
        if vis_img is not None and visualize:
            # 일반 시각화 이미지 저장
            vis_path = os.path.join(vis_dir, f"vis_{img_name}")
            try:
                success = cv2.imwrite(vis_path, vis_img)
                if success:
                    visualization_count += 1
                else:
                    print(f"Warning: Failed to save visualization image: {vis_path}")
            except Exception as e:
                print(f"Error: Failed to save visualization image: {vis_path} - {e}")
            
            # YOLO 변환 결과 시각화 및 비교 이미지 저장
            if compare_img is not None:
                # YOLO 파일에서 바운딩 박스 정보 로드
                yolo_bboxes = load_yolo_bboxes(label_file_path, img_width, img_height)
                
                # YOLO 바운딩 박스 그리기
                compare_img = draw_yolo_bboxes(compare_img, yolo_bboxes, CATEGORY_NAMES)
                
                # 범례 추가
                legend_y = 30
                cv2.putText(compare_img, "JSON Body Boxes (Solid Color)", (10, legend_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                legend_y += 30
                cv2.putText(compare_img, "JSON Face Boxes (Orange)", (10, legend_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, FACE_COLOR, 2)
                
                legend_y += 30
                cv2.putText(compare_img, "YOLO Converted Boxes (Yellow Dotted)", (10, legend_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YOLO_COLOR, 2)
                
                # 비교 이미지 저장
                compare_path = os.path.join(compare_dir, f"compare_{img_name}")
                try:
                    success = cv2.imwrite(compare_path, compare_img)
                    if not success:
                        print(f"Warning: Failed to save comparison image: {compare_path}")
                except Exception as e:
                    print(f"Error: Failed to save comparison image: {compare_path} - {e}")
    
    # 훈련 및 테스트 데이터 구분을 위한 파일 생성
    image_files = [img_data.get("img_name") for img_data in data["scene"]["data"] if "img_name" in img_data]
    
    # 데이터 분할 (80% 훈련, 20% 검증)
    train_size = int(len(image_files) * 0.8)
    train_files = image_files[:train_size]
    valid_files = image_files[train_size:]
    
    # 훈련 데이터 목록 저장
    with open(os.path.join(output_dir, "train.txt"), 'w', encoding='utf-8') as f:
        for img_file in train_files:
            # 이미지 파일의 경로를 지정
            f.write(f"data/images/{img_file}\n")
    
    # 검증 데이터 목록 저장
    with open(os.path.join(output_dir, "valid.txt"), 'w', encoding='utf-8') as f:
        for img_file in valid_files:
            # 이미지 파일의 경로를 지정
            f.write(f"data/images/{img_file}\n")
    
    # YOLO 설정 파일 생성
    yolo_config = f"""classes={len(CATEGORY_NAMES)}
train=data/train.txt
valid=data/valid.txt
names=data/classes.txt
backup=backup/
"""
    
    with open(os.path.join(output_dir, "yolo.data"), 'w', encoding='utf-8') as f:
        f.write(yolo_config)
    
    print(f"Processing complete: {json_file_path} -> {output_dir}")
    if visualize:
        print(f"Visualization images: {visualization_count} created (saved to: {os.path.join(output_dir, 'visualizations')})")
        print(f"Comparison images saved to: {os.path.join(output_dir, 'comparisons')}")
    
    return visualization_count

def find_images_directory(base_dir):
    """
    주어진 디렉토리에서 이미지 파일이 있는 디렉토리 탐색
    
    매개변수:
        base_dir (str): 검색할 기본 디렉토리
    
    반환:
        str: 이미지 디렉토리 경로 또는 None
    """
    # 잠재적인 이미지 디렉토리 패턴
    img_patterns = [
        os.path.join(base_dir, "**", "images", "*.jpg"),
        os.path.join(base_dir, "**", "*.jpg"),
        os.path.join(base_dir, "images", "*.jpg"),
        os.path.join(base_dir, "*.jpg"),
        os.path.join(base_dir, "**", "*.jpeg"),
        os.path.join(base_dir, "**", "*.png")
    ]
    
    for pattern in img_patterns:
        img_files = glob.glob(pattern, recursive=True)
        if img_files:
            # 가장 많은 이미지가 있는 디렉토리 찾기
            dir_counts = {}
            for img_file in img_files:
                img_dir = os.path.dirname(img_file)
                dir_counts[img_dir] = dir_counts.get(img_dir, 0) + 1
            
            # 가장 많은 이미지가 있는 디렉토리 반환
            best_dir = max(dir_counts.items(), key=lambda x: x[1])[0]
            print(f"Automatically found images directory: {best_dir} ({dir_counts[best_dir]} images)")
            return best_dir
    
    print("Warning: No image files found.")
    return None

def process_all_json_files(base_dir, output_base_dir=None, img_base_dir=None, visualize=True):
    """
    지정된 디렉토리 내의 모든 JSON 파일을 처리합니다.
    
    매개변수:
        base_dir (str): 검색할 기본 디렉토리
        output_base_dir (str): 출력 디렉토리 기본 경로 (기본값: None)
        img_base_dir (str): 이미지 검색을 위한 기본 디렉토리 (기본값: None)
        visualize (bool): 바운딩 박스 시각화 여부 (기본값: True)
    """
    # JSON 파일 검색 패턴 (label 디렉토리 내의 모든 .json 파일)
    json_pattern = os.path.join(base_dir, "**", "label", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    # label 디렉토리가 없는 경우 모든 .json 파일 검색
    if not json_files:
        json_pattern = os.path.join(base_dir, "**", "*.json")
        json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        print(f"Warning: No JSON files found in {base_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # 각 JSON 파일 처리
    total_visualization_count = 0
    for i, json_file in enumerate(json_files):
        print(f"\nProcessing ({i+1}/{len(json_files)}): {json_file}")
        
        # 출력 디렉토리 설정
        if output_base_dir:
            # JSON 파일의 상대 경로 계산
            rel_path = os.path.relpath(os.path.dirname(json_file), base_dir)
            
            # 출력 디렉토리 생성
            scene_id = os.path.splitext(os.path.basename(json_file))[0]
            output_dir = os.path.join(output_base_dir, rel_path, f"{scene_id}_yolo_labels")
        else:
            output_dir = None  # 기본값 사용 (JSON 파일과 같은 디렉토리)
        
        # JSON 파일 변환
        try:
            vis_count = convert_json_to_yolo_format(json_file, output_dir, img_base_dir, visualize)
            if vis_count:
                total_visualization_count += vis_count
        except Exception as e:
            print(f"Error: Exception during processing {json_file} - {e}")
    
    print("\nAll files processed!")
    if visualize:
        print(f"Total visualization images created: {total_visualization_count}")
    
    # 카테고리 맵 출력
    print("\nAction Category Map:")
    for category_id, category_name in CATEGORY_NAMES.items():
        print(f"Category {category_id}: {category_name}")
        actions = [f"{action} ({ACTION_TO_ENGLISH[action]})" for action, cat_id in ACTION_CATEGORIES.items() if cat_id == category_id]
        print(f"  Actions: {', '.join(actions)}")


if __name__ == "__main__":
    # 명령줄 인자 처리
    if len(sys.argv) < 2:
        print("Usage: python process_json_with_visualization.py <base_directory> [output_directory] [image_directory] [visualize(y/n)]")
        print("Example: python process_json_with_visualization.py ./sample_data ./yolo_output ./images y")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    img_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 시각화 옵션 처리
    visualize = True
    if len(sys.argv) > 4:
        visualize = sys.argv[4].lower() in ('y', 'yes', 'true', 't', '1')
    
    # 이미지 디렉토리가 없으면 자동으로 찾기 시도
    if visualize and not img_dir:
        img_dir = find_images_directory(base_dir)
        if not img_dir:
            print("Warning: Image directory not found. Visualization may be limited.")
    
    # 모든 JSON 파일 처리
    process_all_json_files(base_dir, output_dir, img_dir, visualize)