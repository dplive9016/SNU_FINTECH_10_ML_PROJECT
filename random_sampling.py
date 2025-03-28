import os
import glob
import random
import shutil
from collections import Counter
import pandas as pd
from tqdm import tqdm

def get_yolo_class_distribution(yolo_dir):
    """
    모든 YOLO txt 파일에서 클래스 ID 분포를 분석합니다.
    
    Args:
        yolo_dir (str): YOLO txt 파일이 위치한 루트 디렉토리
        
    Returns:
        dict: 클래스별 빈도수
        dict: scene_id별 클래스 리스트
    """
    print("YOLO 클래스 분포 분석 중...")
    
    # 모든 *_yolo_labels 디렉토리 찾기
    yolo_pattern = os.path.join(yolo_dir, "**", "*_yolo_labels")
    yolo_dirs = glob.glob(yolo_pattern, recursive=True)
    
    if not yolo_dirs:
        raise ValueError(f"YOLO 라벨 디렉토리를 찾을 수 없습니다: {yolo_pattern}")
    
    print(f"총 {len(yolo_dirs)}개의 YOLO 라벨 디렉토리를 찾았습니다.")
    
    # 모든 클래스 ID 수집
    all_classes = []
    scene_classes = {}  # scene_id별 클래스 리스트 저장
    
    for yolo_dir in tqdm(yolo_dirs):
        # scene_id 추출 (디렉토리 이름에서 _yolo_labels 이전 부분)
        scene_id = os.path.basename(yolo_dir).replace("_yolo_labels", "")
        scene_classes[scene_id] = []
        
        # 디렉토리 내 모든 txt 파일 처리
        txt_files = glob.glob(os.path.join(yolo_dir, "*.txt"))
        
        # classes.txt 파일 제외
        txt_files = [f for f in txt_files if os.path.basename(f) != "classes.txt"]
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.strip():
                        # YOLO 형식: [class_id] [x_center] [y_center] [width] [height]
                        class_id = int(line.strip().split()[0])
                        all_classes.append(class_id)
                        scene_classes[scene_id].append(class_id)
                        
            except Exception as e:
                print(f"오류 발생: {txt_file} 처리 중 - {e}")
    
    # 클래스별 빈도수 계산
    class_counts = Counter(all_classes)
    
    # 클래스명 매핑 (있을 경우)
    class_names = {}
    for yolo_dir in yolo_dirs:
        classes_file = os.path.join(yolo_dir, "classes.txt")
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    class_names[i] = line.strip()
                break  # 하나만 찾으면 됨
            except:
                pass
    
    # 결과 출력
    print("\nYOLO 클래스 분포:")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, f"클래스 {class_id}")
        print(f"{class_id} ({class_name}): {count}개 ({count/len(all_classes)*100:.2f}%)")
    
    return class_counts, scene_classes, class_names

def sample_balanced_scenes(scene_classes, class_counts, sample_ratio=0.25):
    """
    클래스 분포를 고려해 균등하게 scene을 샘플링합니다.
    
    Args:
        scene_classes (dict): scene_id별 클래스 리스트
        class_counts (dict): 전체 클래스별 빈도수
        sample_ratio (float): 샘플링 비율 (0.0 ~ 1.0)
        
    Returns:
        list: 샘플링된 scene_id 리스트
    """
    print(f"\n균등 분포로 {sample_ratio*100:.0f}% 샘플링 중...")
    
    # 클래스별 목표 샘플 수 계산
    total_classes = sum(class_counts.values())
    target_total = int(total_classes * sample_ratio)
    
    # 클래스별 목표 샘플 수 (비율 유지)
    target_per_class = {class_id: max(1, int(count * sample_ratio)) 
                         for class_id, count in class_counts.items()}
    
    # scene별 점수 계산 (희소한 클래스에 더 높은 가중치 부여)
    scene_scores = {}
    class_weights = {class_id: 1.0 / (count + 1) for class_id, count in class_counts.items()}
    
    for scene_id, classes in scene_classes.items():
        # 중복 제거한 클래스별 가중치 합산
        unique_classes = set(classes)
        scene_scores[scene_id] = sum(class_weights.get(class_id, 0) for class_id in unique_classes)
    
    # 점수 기반으로 정렬
    sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 선택된 scene으로부터 클래스 카운팅
    selected_scenes = []
    selected_class_counts = Counter()
    
    # 점수가 높은 scene부터 선택하되, 각 클래스가 목표 수량을 초과하지 않도록 함
    for scene_id, score in sorted_scenes:
        # 이미 충분한 수의 scene을 선택했다면 중단
        if len(selected_scenes) >= len(scene_classes) * sample_ratio:
            break
            
        # 이 scene의 클래스들
        scene_unique_classes = set(scene_classes[scene_id])
        
        # 아직 목표에 도달하지 않은 클래스가 있는지 확인
        has_needed_class = any(
            selected_class_counts[class_id] < target_per_class[class_id]
            for class_id in scene_unique_classes if class_id in target_per_class
        )
        
        if has_needed_class:
            selected_scenes.append(scene_id)
            # 선택된 클래스 카운트 업데이트
            for class_id in scene_classes[scene_id]:
                selected_class_counts[class_id] += 1
    
    # 클래스 분포 확인
    print(f"\n샘플링된 {len(selected_scenes)}개 scene의 클래스 분포:")
    selected_classes = []
    for scene_id in selected_scenes:
        selected_classes.extend(scene_classes[scene_id])
    
    sampled_class_counts = Counter(selected_classes)
    
    # 결과 비교 출력
    print("\n원본 vs 샘플링 클래스 분포 비교:")
    comparison_data = []
    for class_id in sorted(set(class_counts.keys()) | set(sampled_class_counts.keys())):
        original = class_counts.get(class_id, 0)
        sampled = sampled_class_counts.get(class_id, 0)
        original_pct = original / sum(class_counts.values()) * 100 if original else 0
        sampled_pct = sampled / sum(sampled_class_counts.values()) * 100 if sampled else 0
        
        print(f"클래스 {class_id}: 원본 {original}개({original_pct:.2f}%) → 샘플 {sampled}개({sampled_pct:.2f}%)")
        comparison_data.append({
            'class_id': class_id,
            'original_count': original,
            'original_pct': original_pct,
            'sampled_count': sampled,
            'sampled_pct': sampled_pct,
            'ratio_change': sampled_pct / original_pct if original_pct > 0 else float('inf')
        })
    
    # 분포 정보를 DataFrame으로 변환하여 CSV로 저장
    df = pd.DataFrame(comparison_data)
    df.to_csv('class_distribution_comparison.csv', index=False)
    print("\n분포 비교 정보가 'class_distribution_comparison.csv'에 저장되었습니다.")
    
    return selected_scenes

def get_json_paths_from_scene_ids(scene_ids, json_dir):
    """
    scene_id에 해당하는 JSON 파일 경로를 찾습니다.
    
    Args:
        scene_ids (list): scene_id 리스트
        json_dir (str): JSON 파일이 위치한 루트 디렉토리
        
    Returns:
        dict: scene_id를 키로, JSON 파일 경로를 값으로 하는 딕셔너리
    """
    json_paths = {}
    
    for scene_id in tqdm(scene_ids, desc="JSON 파일 검색"):
        json_pattern = os.path.join(json_dir, "**", "label", f"{scene_id}.json")
        json_files = glob.glob(json_pattern, recursive=True)
        
        if json_files:
            json_paths[scene_id] = json_files[0]
    
    return json_paths

def copy_sampled_data(selected_scenes, json_dir, img_dir, yolo_dir, dst_base_dir):
    """
    선택된 scene의 YOLO txt, JSON 파일, 관련 이미지 파일을 복사합니다.
    
    Args:
        selected_scenes (list): 선택된 scene_id 리스트
        json_dir (str): 원본 JSON 파일 디렉토리
        img_dir (str): 원본 이미지 파일 디렉토리
        yolo_dir (str): 원본 YOLO txt 파일 디렉토리
        dst_base_dir (str): 복사될 대상 디렉토리
    """
    print(f"\n선택된 {len(selected_scenes)}개 scene의 파일을 복사하는 중...")
    
    # 대상 디렉토리 생성
    dst_json_dir = os.path.join(dst_base_dir, "라벨링데이터")
    dst_img_dir = os.path.join(dst_base_dir, "원천데이터")
    dst_yolo_dir = os.path.join(dst_base_dir, "yolo_output")
    os.makedirs(dst_json_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_yolo_dir, exist_ok=True)
    
    # 복사 성공/실패 카운트
    json_copied = 0
    img_copied = 0
    yolo_copied = 0
    json_failed = 0
    img_failed = 0
    yolo_failed = 0
    
    # JSON 파일 경로 가져오기
    json_paths = get_json_paths_from_scene_ids(selected_scenes, json_dir)
    
    for scene_id in tqdm(selected_scenes, desc="파일 복사"):
        # 1. YOLO 디렉토리 복사
        yolo_pattern = os.path.join(yolo_dir, "**", f"{scene_id}_yolo_labels")
        yolo_dirs = glob.glob(yolo_pattern, recursive=True)
        
        for yolo_dir_path in yolo_dirs:
            # 상대 경로 계산
            rel_path = os.path.relpath(os.path.dirname(yolo_dir_path), yolo_dir)
            dst_dir = os.path.join(dst_yolo_dir, rel_path)
            
            try:
                # YOLO 디렉토리 전체 복사
                dst_yolo_dir_path = os.path.join(dst_dir, os.path.basename(yolo_dir_path))
                os.makedirs(os.path.dirname(dst_yolo_dir_path), exist_ok=True)
                shutil.copytree(yolo_dir_path, dst_yolo_dir_path)
                
                # txt 파일 개수 카운트
                txt_files = glob.glob(os.path.join(yolo_dir_path, "*.txt"))
                yolo_copied += len(txt_files)
            except Exception as e:
                print(f"YOLO 디렉토리 복사 실패: {yolo_dir_path} - {e}")
                yolo_failed += 1
        
        # 2. JSON 파일 복사 (scene_id에 해당하는 JSON 파일)
        if scene_id in json_paths:
            json_file = json_paths[scene_id]
            rel_path = os.path.relpath(os.path.dirname(json_file), json_dir)
            dst_dir = os.path.join(dst_json_dir, rel_path)
            
            try:
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, os.path.basename(json_file))
                shutil.copy2(json_file, dst_file)
                json_copied += 1
            except Exception as e:
                print(f"JSON 파일 복사 실패: {json_file} - {e}")
                json_failed += 1
        
        # 3. 이미지 파일 복사
        # JSON 파일 경로에서 필요한 패턴 추출
        if scene_id in json_paths:
            json_file = json_paths[scene_id]
            parts = json_file.split(os.sep)
            scene_parts = [p for p in parts if scene_id in p]
            
            if len(scene_parts) >= 2:
                parent_id = None
                # scene_id 바로 앞의 부분을 parent_id로 추출
                for i, part in enumerate(parts):
                    if part == scene_id and i > 0:
                        parent_id = parts[i-1]
                        break
                
                if parent_id:
                    # 이미지 디렉토리 검색 패턴 생성
                    img_dir_pattern = os.path.join(img_dir, "**", parent_id, scene_id, "img")
                    img_dirs = glob.glob(img_dir_pattern, recursive=True)
                    
                    for img_dir_path in img_dirs:
                        # 해당 디렉토리의 모든 이미지 파일
                        img_files = glob.glob(os.path.join(img_dir_path, "*.jpg")) + \
                                  glob.glob(os.path.join(img_dir_path, "*.png")) + \
                                  glob.glob(os.path.join(img_dir_path, "*.jpeg"))
                        
                        # 이미지 복사
                        for img_file in img_files:
                            rel_path = os.path.relpath(os.path.dirname(img_file), img_dir)
                            dst_dir = os.path.join(dst_img_dir, rel_path)
                            
                            try:
                                os.makedirs(dst_dir, exist_ok=True)
                                dst_file = os.path.join(dst_dir, os.path.basename(img_file))
                                shutil.copy2(img_file, dst_file)
                                img_copied += 1
                            except Exception as e:
                                print(f"이미지 파일 복사 실패: {img_file} - {e}")
                                img_failed += 1
    
    print(f"\n복사 완료!")
    print(f"JSON 파일: {json_copied}개 복사 성공, {json_failed}개 실패")
    print(f"이미지 파일: {img_copied}개 복사 성공, {img_failed}개 실패")
    print(f"YOLO txt 파일: {yolo_copied}개 복사 성공, {yolo_failed}개 실패")
    print(f"복사된 파일은 {dst_base_dir} 디렉토리에 저장되었습니다.")

def main():
    # 경로 설정
    base_dir = "/Users/admin/Desktop/Cursor/ML_DRIVER_PROJECT/SNU_FINTECH_10_ML_PROJECT/ML_data"
    json_dir = os.path.join(base_dir, "라벨링데이터")
    img_dir = os.path.join(base_dir, "원천데이터")
    yolo_dir = os.path.join(base_dir, "yolo_output")
    
    # 샘플링된 데이터를 저장할 디렉토리
    output_dir = os.path.join(base_dir, "7_data_rs")
    
    # 샘플링 비율 (25%)
    sample_ratio = 0.25
    
    # YOLO 클래스 분포 분석
    class_counts, scene_classes, class_names = get_yolo_class_distribution(yolo_dir)
    
    # 균등 분포로 scene 샘플링
    selected_scenes = sample_balanced_scenes(scene_classes, class_counts, sample_ratio)
    
    # 선택된 데이터 복사
    copy_sampled_data(selected_scenes, json_dir, img_dir, yolo_dir, output_dir)

if __name__ == "__main__":
    main()