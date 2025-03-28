import os
import shutil
import time
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import re

def build_directory_index(base_dir, pattern, desc="디렉토리 인덱싱"):
    """
    주어진 패턴과 일치하는 경로의 인덱스를 빠르게 구축합니다.
    
    Args:
        base_dir (str): 기본 디렉토리
        pattern (str): 검색 패턴 (정규식)
        
    Returns:
        list: 패턴과 일치하는 경로 리스트
    """
    pattern_regex = re.compile(pattern)
    matched_paths = []
    
    # tqdm을 사용하여 진행 상황 표시
    for root, dirs, files in tqdm(os.walk(base_dir), desc=desc):
        # 패턴에 맞는지 확인
        if pattern_regex.search(root):
            matched_paths.append(root)
    
    return matched_paths

def index_yolo_directories(yolo_dir):
    """
    모든 YOLO 디렉토리의 인덱스를 빠르게 구축합니다.
    
    Args:
        yolo_dir (str): YOLO 디렉토리 기본 경로
        
    Returns:
        dict: scene_id를 키로, YOLO 디렉토리 경로를 값으로 하는 딕셔너리
    """
    print("YOLO 디렉토리 인덱싱 중...")
    start_time = time.time()
    
    # *_yolo_labels 패턴 검색
    yolo_dirs = build_directory_index(yolo_dir, r'_yolo_labels$')
    
    # scene_id를 키로 하는 딕셔너리 생성
    yolo_dir_map = {}
    for dir_path in yolo_dirs:
        dir_name = os.path.basename(dir_path)
        scene_id = dir_name.replace("_yolo_labels", "")
        yolo_dir_map[scene_id] = dir_path
    
    print(f"YOLO 디렉토리 인덱싱 완료: {len(yolo_dir_map)}개 ({time.time()-start_time:.2f}초)")
    return yolo_dir_map

def index_json_files(json_dir):
    """
    모든 JSON 파일의 인덱스를 빠르게 구축합니다.
    
    Args:
        json_dir (str): JSON 파일 기본 경로
        
    Returns:
        dict: scene_id를 키로, JSON 파일 경로를 값으로 하는 딕셔너리
    """
    print("JSON 파일 인덱싱 중...")
    start_time = time.time()
    
    # label 디렉토리를 먼저 찾음
    label_dirs = build_directory_index(json_dir, r'/label$')
    
    # JSON 파일 매핑
    json_file_map = {}
    for label_dir in tqdm(label_dirs, desc="JSON 파일 검색"):
        json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        for json_file in json_files:
            scene_id = os.path.splitext(json_file)[0]
            json_file_map[scene_id] = os.path.join(label_dir, json_file)
    
    print(f"JSON 파일 인덱싱 완료: {len(json_file_map)}개 ({time.time()-start_time:.2f}초)")
    return json_file_map

def index_image_directories(img_dir):
    """
    모든 이미지 디렉토리의 인덱스를 빠르게 구축합니다.
    
    Args:
        img_dir (str): 이미지 파일 기본 경로
        
    Returns:
        dict: scene_id를 키로, 이미지 디렉토리 경로를 값으로 하는 딕셔너리
    """
    print("이미지 디렉토리 인덱싱 중...")
    start_time = time.time()
    
    # img 디렉토리를 찾음
    img_dirs = build_directory_index(img_dir, r'/img$')
    
    # 이미지 디렉토리 매핑 (상위 디렉토리가 scene_id인 경우)
    img_dir_map = {}
    for dir_path in img_dirs:
        parent_dir = os.path.dirname(dir_path)
        scene_id = os.path.basename(parent_dir)
        img_dir_map[scene_id] = dir_path
    
    print(f"이미지 디렉토리 인덱싱 완료: {len(img_dir_map)}개 ({time.time()-start_time:.2f}초)")
    return img_dir_map

def get_yolo_class_distribution(yolo_dir_map):
    """
    모든 YOLO txt 파일에서 클래스 ID 분포를 분석합니다.
    
    Args:
        yolo_dir_map (dict): scene_id를 키로, YOLO 디렉토리 경로를 값으로 하는 딕셔너리
        
    Returns:
        dict: 클래스별 빈도수
        dict: scene_id별 클래스 리스트
    """
    print("YOLO 클래스 분포 분석 중...")
    start_time = time.time()
    
    # 모든 클래스 ID 수집
    all_classes = []
    scene_classes = {}  # scene_id별 클래스 리스트 저장
    
    def process_scene(scene_id, yolo_dir):
        scene_class_list = []
        
        # 디렉토리 내 모든 txt 파일 처리
        txt_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt') and f != 'classes.txt']
        
        for txt_file in txt_files:
            try:
                with open(os.path.join(yolo_dir, txt_file), 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.strip():
                        # YOLO 형식: [class_id] [x_center] [y_center] [width] [height]
                        class_id = int(line.strip().split()[0])
                        scene_class_list.append(class_id)
                        
            except Exception as e:
                print(f"오류 발생: {txt_file} 처리 중 - {e}")
        
        return scene_id, scene_class_list
    
    # 병렬 처리 (최대 8개 워커)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_scene = {
            executor.submit(process_scene, scene_id, yolo_dir): scene_id 
            for scene_id, yolo_dir in yolo_dir_map.items()
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_scene), 
                          total=len(future_to_scene), 
                          desc="클래스 분석"):
            scene_id, class_list = future.result()
            scene_classes[scene_id] = class_list
            all_classes.extend(class_list)
    
    # 클래스별 빈도수 계산
    class_counts = Counter(all_classes)
    
    # 클래스명 매핑 (있을 경우)
    class_names = {}
    for yolo_dir in yolo_dir_map.values():
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
    print(f"YOLO 클래스 분포 분석 완료 ({time.time()-start_time:.2f}초)")
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
    start_time = time.time()
    
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
    selected_classes = []
    for scene_id in selected_scenes:
        selected_classes.extend(scene_classes[scene_id])
    
    sampled_class_counts = Counter(selected_classes)
    
    # 결과 비교 출력
    print(f"샘플링 완료: {len(selected_scenes)}개 scene ({time.time()-start_time:.2f}초)")
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

def copy_files_parallel(src_files, dst_base_dir, desc="파일 복사"):
    """
    파일 목록을 병렬로 복사합니다.
    
    Args:
        src_files (list): (원본 파일 경로, 대상 파일 경로) 튜플 리스트
        dst_base_dir (str): 대상 기본 디렉토리
        desc (str): 진행 상황 설명
    
    Returns:
        tuple: (성공 카운트, 실패 카운트)
    """
    success_count = 0
    failed_count = 0
    
    def copy_file(src, dst):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"파일 복사 실패: {src} -> {dst} - {e}")
            return False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(copy_file, src, dst) for src, dst in src_files]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc=desc):
            if future.result():
                success_count += 1
            else:
                failed_count += 1
    
    return success_count, failed_count

def copy_sampled_data(selected_scenes, json_file_map, img_dir_map, yolo_dir_map, dst_base_dir):
    """
    선택된 scene의 YOLO txt, JSON 파일, 관련 이미지 파일을 복사합니다.
    
    Args:
        selected_scenes (list): 선택된 scene_id 리스트
        json_file_map (dict): scene_id를 키로, JSON 파일 경로를 값으로 하는 딕셔너리
        img_dir_map (dict): scene_id를 키로, 이미지 디렉토리 경로를 값으로 하는 딕셔너리
        yolo_dir_map (dict): scene_id를 키로, YOLO 디렉토리 경로를 값으로 하는 딕셔너리
        dst_base_dir (str): 복사될 대상 디렉토리
    """
    print(f"\n선택된 {len(selected_scenes)}개 scene의 파일을 복사하는 중...")
    start_time = time.time()
    
    # 대상 디렉토리 생성
    dst_json_dir = os.path.join(dst_base_dir, "라벨링데이터")
    dst_img_dir = os.path.join(dst_base_dir, "원천데이터")
    dst_yolo_dir = os.path.join(dst_base_dir, "yolo_output")
    os.makedirs(dst_json_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_yolo_dir, exist_ok=True)
    
    # 복사할 파일 목록 준비
    json_files_to_copy = []
    img_files_to_copy = []
    yolo_dirs_to_copy = []
    
    # 1. JSON 파일 목록 준비
    for scene_id in selected_scenes:
        if scene_id in json_file_map:
            src_json = json_file_map[scene_id]
            rel_path = os.path.relpath(src_json, os.path.dirname(os.path.dirname(json_file_map[scene_id])))
            dst_json = os.path.join(dst_json_dir, rel_path)
            json_files_to_copy.append((src_json, dst_json))
    
    # 2. 이미지 파일 목록 준비
    for scene_id in selected_scenes:
        if scene_id in img_dir_map:
            img_dir = img_dir_map[scene_id]
            img_files = [f for f in os.listdir(img_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in img_files:
                src_img = os.path.join(img_dir, img_file)
                rel_path = os.path.relpath(src_img, os.path.dirname(os.path.dirname(img_dir)))
                dst_img = os.path.join(dst_img_dir, rel_path)
                img_files_to_copy.append((src_img, dst_img))
    
    # 3. YOLO 디렉토리 복사 준비
    for scene_id in selected_scenes:
        if scene_id in yolo_dir_map:
            yolo_dir = yolo_dir_map[scene_id]
            yolo_dirs_to_copy.append((scene_id, yolo_dir))
    
    # 4. JSON 파일 복사
    print(f"JSON 파일 {len(json_files_to_copy)}개 복사 시작...")
    json_success, json_failed = copy_files_parallel(
        json_files_to_copy, dst_base_dir, "JSON 파일 복사")
    
    # 5. 이미지 파일 복사
    print(f"이미지 파일 {len(img_files_to_copy)}개 복사 시작...")
    img_success, img_failed = copy_files_parallel(
        img_files_to_copy, dst_base_dir, "이미지 파일 복사")
    
    # 6. YOLO 디렉토리 복사
    print(f"YOLO 디렉토리 {len(yolo_dirs_to_copy)}개 복사 시작...")
    yolo_copied = 0
    yolo_failed = 0
    
    # YOLO 디렉토리 복사를 위한 파일 목록 수집
    yolo_files_to_copy = []
    for scene_id, yolo_dir in tqdm(yolo_dirs_to_copy, desc="YOLO 파일 준비"):
        rel_dir = os.path.relpath(os.path.dirname(yolo_dir), os.path.dirname(os.path.dirname(yolo_dir)))
        dst_dir = os.path.join(dst_yolo_dir, rel_dir, os.path.basename(yolo_dir))
        
        # 디렉토리 내 모든 파일
        for file_name in os.listdir(yolo_dir):
            src_file = os.path.join(yolo_dir, file_name)
            if os.path.isfile(src_file):  # 파일만 복사
                dst_file = os.path.join(dst_dir, file_name)
                yolo_files_to_copy.append((src_file, dst_file))
    
    # YOLO 파일 복사
    yolo_success, yolo_failed = copy_files_parallel(
        yolo_files_to_copy, dst_base_dir, "YOLO 파일 복사")
    
    print(f"\n복사 완료! ({time.time()-start_time:.2f}초)")
    print(f"JSON 파일: {json_success}개 복사 성공, {json_failed}개 실패")
    print(f"이미지 파일: {img_success}개 복사 성공, {img_failed}개 실패")
    print(f"YOLO 파일: {yolo_success}개 복사 성공, {yolo_failed}개 실패")
    print(f"복사된 파일은 {dst_base_dir} 디렉토리에 저장되었습니다.")

def main():
    # 경로 설정
    base_dir = "/Users/admin/Desktop/Cursor/ML_DRIVER_PROJECT/SNU_FINTECH_10_ML_PROJECT/ML_data_V1"
    json_dir = os.path.join(base_dir, "라벨링데이터")
    img_dir = os.path.join(base_dir, "원천데이터")
    yolo_dir = os.path.join(base_dir, "yolo_output")
    
    # 샘플링된 데이터를 저장할 디렉토리
    output_dir = os.path.join(base_dir, "ML_data_25pct_V1")
    
    # 샘플링 비율 (25%)
    sample_ratio = 0.25
    
    # 시작 시간 기록
    total_start_time = time.time()
    
    # 1. 디렉토리 인덱싱 (빠른 검색을 위한 사전 작업)
    yolo_dir_map = index_yolo_directories(yolo_dir)
    json_file_map = index_json_files(json_dir)
    img_dir_map = index_image_directories(img_dir)
    
    # 2. YOLO 클래스 분포 분석
    class_counts, scene_classes, class_names = get_yolo_class_distribution(yolo_dir_map)
    
    # 3. 균등 분포로 scene 샘플링
    selected_scenes = sample_balanced_scenes(scene_classes, class_counts, sample_ratio)
    
    # 4. 선택된 데이터 복사
    copy_sampled_data(selected_scenes, json_file_map, img_dir_map, yolo_dir_map, output_dir)
    
    # 종료 시간 기록
    print(f"\n총 실행 시간: {time.time() - total_start_time:.2f}초")

if __name__ == "__main__":
    main()