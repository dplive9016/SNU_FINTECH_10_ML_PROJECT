import os
import sys
import cv2
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rc('font', family='AppleGothic')

# 카테고리 이름 정의
CATEGORY_NAMES = {
    0: "졸음/피로 및 무기력",
    1: "운전자 방해 및 위험 행동",
    2: "물건 사용/조작 관련",
    3: "신체 동작 및 접촉/상호작용",
}

# 카테고리별 색상 정의 (RGB 형식)
CATEGORY_COLORS = {
    0: (1, 0, 0),    # 빨간색 (졸음/피로)
    1: (0, 1, 0),    # 녹색 (운전자 방해)
    2: (0, 0, 1),    # 파란색 (물건 사용)
    3: (0, 1, 1),    # 청록색 (신체 동작)
}

def find_directories(base_dir):
    """
    기본 디렉토리에서 라벨링데이터, 원천데이터, yolo_output 디렉토리를 찾습니다.
    
    매개변수:
        base_dir (str): 기본 디렉토리 경로
    
    반환값:
        tuple: (label_dir, image_dir, yolo_dir) 경로
    """
    # 디렉토리 경로 초기화
    label_dir = None
    image_dir = None
    yolo_dir = None
    
    # 기본 디렉토리 내 모든 디렉토리 탐색
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            
            # '라벨링데이터' 디렉토리 찾기
            if dir_name == "라벨링데이터" or "label" in dir_name.lower():
                label_dir = full_path
            
            # '원천데이터' 디렉토리 찾기
            elif dir_name == "원천데이터" or "source" in dir_name.lower() or "data" in dir_name.lower():
                image_dir = full_path
            
            # 'yolo_output' 디렉토리 찾기
            elif "yolo" in dir_name.lower() or "output" in dir_name.lower():
                yolo_dir = full_path
        
        # 충분한 깊이까지만 탐색 (기본 디렉토리의 직계 자식들만)
        if root == base_dir:
            break
    
    # 수동으로 경로 구성 (자동 찾기 실패 시)
    if label_dir is None:
        potential_label_dir = os.path.join(base_dir, "라벨링데이터")
        if os.path.exists(potential_label_dir):
            label_dir = potential_label_dir
    
    if image_dir is None:
        potential_image_dir = os.path.join(base_dir, "원천데이터")
        if os.path.exists(potential_image_dir):
            image_dir = potential_image_dir
    
    if yolo_dir is None:
        potential_yolo_dir = os.path.join(base_dir, "yolo_output")
        if os.path.exists(potential_yolo_dir):
            yolo_dir = potential_yolo_dir
    
    return label_dir, image_dir, yolo_dir

def find_json_files(label_dir, max_files=None, scene_id=None):
    """
    라벨 디렉토리에서 JSON 파일을 찾습니다.
    
    매개변수:
        label_dir (str): 라벨 디렉토리 경로
        max_files (int): 최대 파일 수 (기본값: None, 제한 없음)
        scene_id (str): 특정 장면 ID (기본값: None, 모든 장면)
    
    반환값:
        list: 찾은 JSON 파일 경로 리스트
    """
    if scene_id:
        # 특정 장면 ID 검색
        json_pattern = os.path.join(label_dir, "**", f"{scene_id}.json")
        json_files = glob.glob(json_pattern, recursive=True)
        
        # 디렉토리 구조 깊게 탐색 (label 하위 디렉토리까지)
        if not json_files:
            json_pattern = os.path.join(label_dir, "**", "label", f"{scene_id}.json")
            json_files = glob.glob(json_pattern, recursive=True)
    else:
        # 모든 JSON 파일 검색
        json_pattern = os.path.join(label_dir, "**", "*.json")
        all_json_files = glob.glob(json_pattern, recursive=True)
        
        # label 디렉토리에 있는 파일만 필터링
        json_files = [f for f in all_json_files if "label" in os.path.dirname(f)]
        
        # label 디렉토리가 없으면 모든 JSON 파일 사용
        if not json_files:
            json_files = all_json_files
    
    # 최대 파일 수 제한
    if max_files and len(json_files) > max_files:
        json_files = json_files[:max_files]
    
    return json_files

def find_image_for_json(json_path, image_dir):
    """
    JSON 파일에 해당하는 이미지 디렉토리를 찾습니다.
    
    매개변수:
        json_path (str): JSON 파일 경로
        image_dir (str): 이미지 기본 디렉토리
    
    반환값:
        str: 이미지 디렉토리 경로
    """
    # JSON 파일명에서 장면 ID 추출
    scene_id = os.path.splitext(os.path.basename(json_path))[0]
    video_id = scene_id[:10] if len(scene_id) >= 10 else scene_id
    
    # 가능한 이미지 디렉토리 구조들
    possible_paths = [
        # 원천데이터/TS1/SGA2100300/SGA2100300S0001/img/
        os.path.join(image_dir, "**", video_id, scene_id, "img"),
        # 원천데이터/SGA2100300/SGA2100300S0001/img/
        os.path.join(image_dir, video_id, scene_id, "img"),
        # 기타 가능한 구조들
        os.path.join(image_dir, "**", scene_id, "img"),
        os.path.join(image_dir, "**", scene_id, "images"),
        os.path.join(image_dir, "**", scene_id)
    ]
    
    # 각 가능한 경로 확인
    for path_pattern in possible_paths:
        matching_dirs = glob.glob(path_pattern, recursive=True)
        if matching_dirs:
            return matching_dirs[0]
    
    # 이미지 디렉토리를 찾지 못한 경우, JSON 파일에서 직접 이미지 찾기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if "scene" in json_data and "data" in json_data["scene"]:
            # 첫 번째 이미지 이름 가져오기
            first_img = json_data["scene"]["data"][0].get("img_name", "")
            if first_img:
                # 이미지 파일 검색
                img_pattern = os.path.join(image_dir, "**", first_img)
                matching_images = glob.glob(img_pattern, recursive=True)
                if matching_images:
                    return os.path.dirname(matching_images[0])
    except:
        pass
    
    return None

def find_yolo_dir_for_json(json_path, yolo_dir):
    """
    JSON 파일에 해당하는 YOLO 라벨 디렉토리를 찾습니다.
    
    매개변수:
        json_path (str): JSON 파일 경로
        yolo_dir (str): YOLO 기본 디렉토리
    
    반환값:
        str: YOLO 라벨 디렉토리 경로
    """
    # JSON 파일명에서 장면 ID 추출
    scene_id = os.path.splitext(os.path.basename(json_path))[0]
    video_id = scene_id[:10] if len(scene_id) >= 10 else scene_id
    
    # 가능한 YOLO 디렉토리 구조들
    possible_paths = [
        # yolo_output/TL1/SGA2100300/SGA2100300S0001/label/SGA2100300S0001_yolo_labels/
        os.path.join(yolo_dir, "**", video_id, scene_id, "**", f"{scene_id}_yolo_labels"),
        # yolo_output/SGA2100300/SGA2100300S0001/label/SGA2100300S0001_yolo_labels/
        os.path.join(yolo_dir, video_id, scene_id, "**", f"{scene_id}_yolo_labels"),
        # 기타 가능한 구조들
        os.path.join(yolo_dir, "**", f"{scene_id}_yolo_labels"),
        os.path.join(yolo_dir, "**", scene_id, "labels"),
        os.path.join(yolo_dir, "**", scene_id)
    ]
    
    # 각 가능한 경로 확인
    for path_pattern in possible_paths:
        matching_dirs = glob.glob(path_pattern, recursive=True)
        if matching_dirs:
            return matching_dirs[0]
    
    return None

def visualize_bounding_boxes(json_path, img_dir, yolo_dir, output_dir=None, show_plot=True):
    """
    JSON 파일, 이미지, YOLO 라벨을 사용하여 바운딩 박스를 시각화합니다.
    
    매개변수:
        json_path (str): JSON 파일 경로
        img_dir (str): 이미지 디렉토리 경로
        yolo_dir (str): YOLO 라벨 디렉토리 경로
        output_dir (str): 결과 저장 디렉토리 (기본값: None)
        show_plot (bool): 시각화 결과를 화면에 표시할지 여부 (기본값: True)
    """
    # JSON 파일 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"JSON 파일을 읽을 수 없습니다: {json_path}, 오류: {e}")
        return
    
    # 장면 ID 추출
    scene_id = os.path.splitext(os.path.basename(json_path))[0]
    
    # 출력 디렉토리 생성
    if output_dir:
        results_dir = os.path.join(output_dir, scene_id)
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None
    
    # JSON 구조 검증
    if "scene" not in json_data or "data" not in json_data["scene"]:
        print(f"유효하지 않은 JSON 구조: {json_path}")
        return
    
    processed_count = 0
    
    # 각 이미지에 대해 처리
    for img_data in json_data["scene"]["data"]:
        if "img_name" not in img_data:
            continue
        
        img_name = img_data["img_name"]
        image_path = os.path.join(img_dir, img_name)
        
        # 이미지 파일이 존재하는지 확인
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            continue
        
        # YOLO 라벨 파일 경로
        base_name = os.path.splitext(img_name)[0]
        yolo_path = os.path.join(yolo_dir, f"{base_name}.txt")
        
        # YOLO 라벨 파일이 존재하는지 확인
        if not os.path.exists(yolo_path):
            print(f"YOLO 라벨 파일을 찾을 수 없습니다: {yolo_path}")
            continue
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            continue
        
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기
        img_height, img_width = image.shape[:2]
        
        # 그림 설정
        plt.figure(figsize=(15, 10))
        
        # 원본 이미지에 JSON 바운딩 박스 그리기 (왼쪽)
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f'JSON 바운딩 박스: {img_name}')
        plt.axis('off')
        ax1 = plt.gca()
        
        # JSON에서 바운딩 박스 정보 추출 및 그리기
        if "occupant" in img_data:
            for occupant in img_data["occupant"]:
                action = occupant.get("action", "알 수 없음")
                
                # 바디 바운딩 박스
                if "body_b_box" in occupant:
                    x1, y1, x2, y2 = occupant["body_b_box"]
                    width = x2 - x1
                    height = y2 - y1
                    
                    rect = Rectangle((x1, y1), width, height, 
                                    linewidth=2, edgecolor='r', facecolor='none')
                    ax1.add_patch(rect)
                    plt.text(x1, y1 - 5, f'Body: {action}', 
                            color='r', fontsize=9, weight='bold')
                
                # 페이스 바운딩 박스
                if "face_b_box" in occupant:
                    x, y, width, height = occupant["face_b_box"]
                    
                    rect = Rectangle((x, y), width, height, 
                                    linewidth=2, edgecolor='g', facecolor='none')
                    ax1.add_patch(rect)
                    plt.text(x, y - 5, f'Face: {action}', 
                            color='g', fontsize=9, weight='bold')
        
        # YOLO 레이블 파일 읽기 및 시각화 (오른쪽)
        plt.subplot(1, 2, 2)
        plt.imshow(image_rgb)
        plt.title(f'YOLO 바운딩 박스: {base_name}.txt')
        plt.axis('off')
        ax2 = plt.gca()
        
        # YOLO 레이블 파일 읽기
        try:
            with open(yolo_path, 'r') as f:
                lines = f.read().strip().split('\n')
            
            # 각 바운딩 박스 그리기
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                category_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 박스의 좌상단 좌표 계산
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                
                # 박스 타입 결정 (짝수 인덱스는 바디, 홀수 인덱스는 페이스로 가정)
                box_type = "Body" if i % 2 == 0 else "Face"
                color = 'r' if box_type == "Body" else 'g'
                
                # 박스 그리기
                rect = Rectangle((x_min, y_min), width, height, 
                                linewidth=2, edgecolor=color, facecolor='none')
                ax2.add_patch(rect)
                
                # 레이블 표시
                category_name = CATEGORY_NAMES.get(category_id, f"카테고리 {category_id}")
                plt.text(x_min, y_min - 5, f'{box_type}: {category_name}', 
                        color=color, fontsize=9, weight='bold')
        except Exception as e:
            print(f"YOLO 레이블 파일을 읽을 수 없습니다: {yolo_path}, 오류: {e}")
            continue
        
        # 결과 저장 또는 표시
        plt.tight_layout()
        if results_dir:
            result_path = os.path.join(results_dir, f"{base_name}_comparison.png")
            plt.savefig(result_path, dpi=150)
            print(f"결과 이미지 저장 완료: {result_path}")
        
        if show_plot:
            plt.show()
            input("다음 이미지를 보려면 Enter 키를 누르세요...")
        
        plt.close()
        processed_count += 1
    
    return processed_count

def main():
    """
    메인 함수: 명령줄 인자를 처리하고 바운딩 박스 시각화를 실행합니다.
    """
    parser = argparse.ArgumentParser(description='간편한 바운딩 박스 시각화 도구')
    parser.add_argument('base_dir', help='기본 디렉토리 경로 (예: ./sample_data)')
    parser.add_argument('--output', '-o', help='결과 저장 디렉토리')
    parser.add_argument('--scene', '-s', help='처리할 특정 장면 ID')
    parser.add_argument('--limit', '-l', type=int, default=5, help='처리할 최대 장면 수 (기본값: 5)')
    parser.add_argument('--no-show', action='store_true', help='결과를 화면에 표시하지 않음')
    
    args = parser.parse_args()
    
    # 기본 디렉토리 확인
    if not os.path.exists(args.base_dir):
        print(f"기본 디렉토리가 존재하지 않습니다: {args.base_dir}")
        return
    
    # 필요한 디렉토리 찾기
    print(f"디렉토리 구조 분석 중... {args.base_dir}")
    label_dir, image_dir, yolo_dir = find_directories(args.base_dir)
    
    if not label_dir:
        print("라벨링 데이터 디렉토리를 찾을 수 없습니다.")
        print("다음 디렉토리를 수동으로 지정해주세요: '라벨링데이터' 또는 이와 유사한 디렉토리")
        return
    
    if not image_dir:
        print("원천 데이터 디렉토리를 찾을 수 없습니다.")
        print("다음 디렉토리를 수동으로 지정해주세요: '원천데이터' 또는 이와 유사한 디렉토리")
        return
    
    if not yolo_dir:
        print("YOLO 출력 디렉토리를 찾을 수 없습니다.")
        print("다음 디렉토리를 수동으로 지정해주세요: 'yolo_output' 또는 이와 유사한 디렉토리")
        return
    
    print(f"라벨링 데이터 디렉토리: {label_dir}")
    print(f"원천 데이터 디렉토리: {image_dir}")
    print(f"YOLO 출력 디렉토리: {yolo_dir}")
    
    # JSON 파일 찾기
    json_files = find_json_files(label_dir, args.limit, args.scene)
    
    if not json_files:
        print(f"JSON 파일을 찾을 수 없습니다.")
        if args.scene:
            print(f"장면 ID '{args.scene}'에 해당하는 JSON 파일이 없습니다.")
        return
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # 출력 디렉토리 생성
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # 각 JSON 파일에 대해 처리
    total_processed = 0
    for i, json_path in enumerate(json_files):
        print(f"\n[{i+1}/{len(json_files)}] 처리 중: {json_path}")
        
        # 장면 ID 추출
        scene_id = os.path.splitext(os.path.basename(json_path))[0]
        
        # 이미지 디렉토리 찾기
        img_dir = find_image_for_json(json_path, image_dir)
        if not img_dir:
            print(f"이미지 디렉토리를 찾을 수 없습니다: {scene_id}")
            continue
        
        # YOLO 디렉토리 찾기
        yolo_label_dir = find_yolo_dir_for_json(json_path, yolo_dir)
        if not yolo_label_dir:
            print(f"YOLO 라벨 디렉토리를 찾을 수 없습니다: {scene_id}")
            continue
        
        print(f"이미지 디렉토리: {img_dir}")
        print(f"YOLO 라벨 디렉토리: {yolo_label_dir}")
        
        # 바운딩 박스 시각화
        processed = visualize_bounding_boxes(
            json_path, img_dir, yolo_label_dir, 
            args.output, not args.no_show
        )
        
        total_processed += processed
    
    print(f"\n총 {len(json_files)}개 장면, {total_processed}개 이미지 처리 완료!")

if __name__ == "__main__":
    main()