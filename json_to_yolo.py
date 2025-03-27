import os
import glob
import sys
import json

# 액션 카테고리 정의
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

# 카테고리 이름 정의
CATEGORY_NAMES = {
    0: "졸음/피로 및 무기력",
    1: "운전자 방해 및 위험 행동",
    2: "물건 사용/조작 관련",
    3: "신체 동작 및 접촉/상호작용",
}

def convert_json_to_yolo_format(json_file_path, output_dir=None):
    """
    JSON 파일을 YOLO v4 형식으로 변환합니다.
    
    매개변수:
        json_file_path (str): JSON 파일 경로
        output_dir (str): 출력 디렉토리 (기본값: None, 자동 생성)
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
    for img_data in data["scene"]["data"]:
        if "img_name" not in img_data or "occupant" not in img_data:
            continue  # 필요한 필드가 없으면 건너뜀
            
        img_name = img_data["img_name"]
        label_file_name = os.path.splitext(img_name)[0] + ".txt"
        label_file_path = os.path.join(output_dir, label_file_name)
        
        with open(label_file_path, 'w', encoding='utf-8') as f:
            for occupant in img_data["occupant"]:
                # 바운딩 박스와 액션 정보 확인
                if "action" not in occupant:
                    continue
                
                action = occupant["action"]
                # 카테고리 매핑 - 맵에 없는 액션은 기본값 3(신체 동작)으로 처리
                category_id = ACTION_CATEGORIES.get(action, 3)
                
                # 이미지 크기는 JSON에 명시되어 있지 않으므로 가정 
                # (샘플 JSON 분석 결과 약 1280x720으로 추정)
                img_width, img_height = 1280, 720
                
                # 1. body_b_box 처리
                if "body_b_box" in occupant:
                    try:
                        x1, y1, x2, y2 = occupant["body_b_box"]
                        
                        # YOLO 형식으로 변환: [class_id] [x_center] [y_center] [width] [height]
                        box_width = (x2 - x1) / img_width
                        box_height = (y2 - y1) / img_height
                        x_center = (x1 + (x2 - x1) / 2) / img_width
                        y_center = (y1 + (y2 - y1) / 2) / img_height
                        
                        # YOLO 형식으로 저장
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    except Exception as e:
                        print(f"Error processing body bounding box in {img_name}: {e}")
                
                # 2. face_b_box 처리 - 같은 카테고리로 저장
                if "face_b_box" in occupant:
                    try:
                        # face_b_box는 [x, y, width, height] 형식으로 저장되어 있음
                        x, y, width, height = occupant["face_b_box"]
                        
                        # YOLO 형식으로 변환
                        # face_b_box의 중심 계산
                        x_center = (x + width / 2) / img_width
                        y_center = (y + height / 2) / img_height
                        box_width = width / img_width
                        box_height = height / img_height
                        
                        # YOLO 형식으로 저장 (같은 카테고리 ID 사용)
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    except Exception as e:
                        print(f"Error processing face bounding box in {img_name}: {e}")
    
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
    
    print(f"처리 완료: {json_file_path} -> {output_dir}")


def process_all_json_files(base_dir, output_base_dir=None):
    """
    지정된 디렉토리 내의 모든 JSON 파일을 처리합니다.
    
    매개변수:
        base_dir (str): 검색할 기본 디렉토리
        output_base_dir (str): 출력 디렉토리 기본 경로 (기본값: None)
    """
    # JSON 파일 검색 패턴 (label 디렉토리 내의 모든 .json 파일)
    json_pattern = os.path.join(base_dir, "**", "label", "*.json")
    
    # 모든 JSON 파일 찾기
    json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        print(f"경고: {json_pattern} 경로에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # 각 JSON 파일 처리
    for i, json_file in enumerate(json_files):
        print(f"처리 중 ({i+1}/{len(json_files)}): {json_file}")
        
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
            convert_json_to_yolo_format(json_file, output_dir)
        except Exception as e:
            print(f"오류 발생: {json_file} 처리 중 예외 - {e}")


if __name__ == "__main__":
    # 명령줄 인자 처리
    if len(sys.argv) < 2:
        print("사용법: python json_to_yolo.py <base_directory> [output_directory]")
        print("예시: python json_to_yolo.py ./sample_data/라벨링데이터 ./sample_data/yolo_output")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 모든 JSON 파일 처리
    process_all_json_files(base_dir, output_dir)
    print("모든 파일 처리 완료!")
    
    # 카테고리 맵 출력
    print("\n액션 카테고리 맵:")
    for category_id, category_name in CATEGORY_NAMES.items():
        print(f"카테고리 {category_id}: {category_name}")
        actions = [action for action, cat_id in ACTION_CATEGORIES.items() if cat_id == category_id]
        print(f"  액션: {', '.join(actions)}")