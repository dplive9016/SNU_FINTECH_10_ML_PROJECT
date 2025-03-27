import os
import glob
import shutil
import yaml
import random
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rc('font', family='AppleGothic')

def prepare_data_for_yolov8(
    project_dir,
    image_dir,
    label_dir,
    output_dir="yolov8_dataset", 
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
):
    """
    YOLOv8 학습을 위해 데이터셋을 준비합니다.
    
    Args:
        project_dir (str): 프로젝트 루트 디렉토리
        image_dir (str): 이미지 파일이 위치한 디렉토리
        label_dir (str): YOLO 형식 라벨 파일이 위치한 디렉토리
        output_dir (str): 출력 데이터셋 디렉토리
        train_ratio (float): 훈련 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
    """
    # 절대 경로 변환
    project_dir = os.path.abspath(project_dir)
    image_dir = os.path.abspath(image_dir)
    label_dir = os.path.abspath(label_dir)
    output_dir = os.path.join(project_dir, output_dir)
    
    # 출력 디렉토리 구조 생성
    os.makedirs(output_dir, exist_ok=True)
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "labels"), exist_ok=True)
    
    # 클래스 이름 정의
    class_names = [
        "졸음/피로 및 무기력",
        "운전자 방해 및 위험 행동",
        "물건 사용/조작 관련",
        "신체 동작 및 접촉/상호작용"
    ]
    
    # 클래스 매핑 파일 생성
    with open(os.path.join(output_dir, "classes.txt"), 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # 원본 이미지 파일 검색 (재귀적으로 모든 img 디렉토리 아래의 jpg 파일 검색)
    image_pattern = os.path.join(image_dir, "**", "img", "*.jpg")
    image_files = glob.glob(image_pattern, recursive=True)
    
    if not image_files:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_pattern}")
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    
    # 라벨 파일 디렉토리 검색 (재귀적으로 모든 *_yolo_labels 디렉토리 검색)
    label_pattern = os.path.join(label_dir, "**", "*_yolo_labels")
    label_dirs = glob.glob(label_pattern, recursive=True)
    
    if not label_dirs:
        raise FileNotFoundError(f"라벨 디렉토리를 찾을 수 없습니다: {label_pattern}")
    
    print(f"총 {len(label_dirs)}개의 라벨 디렉토리를 찾았습니다.")
    
    # 이미지-라벨 쌍 매칭
    valid_pairs = []
    
    for img_path in tqdm(image_files, desc="이미지-라벨 매칭"):
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        
        # 해당 이미지의 라벨 파일 찾기
        label_found = False
        for label_dir in label_dirs:
            label_path = os.path.join(label_dir, f"{img_base}.txt")
            if os.path.exists(label_path):
                valid_pairs.append((img_path, label_path))
                label_found = True
                break
        
        if not label_found:
            print(f"경고: {img_name}에 대한 라벨 파일을 찾을 수 없습니다.")
    
    print(f"총 {len(valid_pairs)}개의 유효한 이미지-라벨 쌍을 찾았습니다.")
    
    # 데이터셋 분할
    random.shuffle(valid_pairs)
    total_samples = len(valid_pairs)
    
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count+val_count]
    test_pairs = valid_pairs[train_count+val_count:]
    
    print(f"훈련 데이터: {len(train_pairs)}개")
    print(f"검증 데이터: {len(val_pairs)}개")
    print(f"테스트 데이터: {len(test_pairs)}개")
    
    # 데이터셋 복사
    dataset_splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }
    
    for split_name, pairs in dataset_splits.items():
        for i, (img_path, label_path) in enumerate(tqdm(pairs, desc=f"{split_name} 데이터 복사")):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            
            # 이미지 복사
            dst_img_path = os.path.join(output_dir, split_name, "images", img_name)
            shutil.copy2(img_path, dst_img_path)
            
            # 라벨 복사
            dst_label_path = os.path.join(output_dir, split_name, "labels", label_name)
            shutil.copy2(label_path, dst_label_path)
    
    # YOLOv8 설정 파일 생성
    yaml_config = {
        "path": output_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names
    }
    
    with open(os.path.join(output_dir, "dataset.yaml"), 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"YOLOv8 데이터셋이 {output_dir}에 준비되었습니다.")
    return output_dir


def train_yolov8(dataset_yaml, output_dir, model_size="n", epochs=100, batch_size=16, img_size=640):
    """
    YOLOv8 모델 학습을 실행합니다.
    
    Args:
        dataset_yaml (str): 데이터셋 YAML 파일 경로
        output_dir (str): 출력 디렉토리
        model_size (str): 모델 크기 (n, s, m, l, x)
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기
        img_size (int): 입력 이미지 크기
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics 패키지가 설치되어 있지 않습니다. 설치 중...")
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics"])
        from ultralytics import YOLO
    
    # 현재 날짜 및 시간으로 저장 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, "models", f"train_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 로드
    model = YOLO(f"yolov8{model_size}.pt")
    
    # 학습 실행 - 명시적으로 프로젝트 디렉토리와 이름 지정
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        plots=True,
        save=True,
        project=os.path.join(output_dir, "runs"),
        name=f"train_{timestamp}",
        device='mps'  # MPS 장치 사용 설정
    )
    
    # 학습된 가중치 명시적으로 저장
    try:
        # best.pt 모델 저장
        best_model_path = os.path.join(output_dir, "runs", f"train_{timestamp}", "weights", "best.pt")
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, os.path.join(model_dir, "best.pt"))
            print(f"최고 성능 모델 저장됨: {os.path.join(model_dir, 'best.pt')}")
        
        # last.pt 모델 저장
        last_model_path = os.path.join(output_dir, "runs", f"train_{timestamp}", "weights", "last.pt")
        if os.path.exists(last_model_path):
            shutil.copy2(last_model_path, os.path.join(model_dir, "last.pt"))
            print(f"마지막 모델 저장됨: {os.path.join(model_dir, 'last.pt')}")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
    
    print(f"학습 완료. 모델이 {model_dir}에 저장되었습니다.")
    return model


def validate_yolov8(model, dataset_yaml, output_dir):
    """
    YOLOv8 모델 검증을 실행합니다.
    
    Args:
        model: 학습된 YOLOv8 모델
        dataset_yaml (str): 데이터셋 YAML 파일 경로
        output_dir (str): 출력 디렉토리
    """
    # 현재 날짜 및 시간으로 저장 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    val_dir = os.path.join(output_dir, "validation", f"val_{timestamp}")
    os.makedirs(val_dir, exist_ok=True)
    
    # 모델 검증 - 명시적으로 프로젝트 디렉토리와 이름 지정
    results = model.val(
        data=dataset_yaml,
        project=os.path.join(output_dir, "runs"),
        name=f"val_{timestamp}",
        device='mps'  # MPS 장치 사용 설정
    )
    
    # 검증 결과 저장
    try:
        val_result_dir = os.path.join(output_dir, "runs", f"val_{timestamp}")
        if os.path.exists(val_result_dir):
            # 결과 파일 복사
            for file in os.listdir(val_result_dir):
                if file.endswith(".png") or file.endswith(".json"):
                    shutil.copy2(os.path.join(val_result_dir, file), os.path.join(val_dir, file))
            print(f"검증 결과가 {val_dir}에 저장되었습니다.")
    except Exception as e:
        print(f"검증 결과 저장 중 오류 발생: {e}")
    
    print(f"검증 완료. 결과: {results}")
    return results


def test_yolov8_with_visualization(model, test_images_dir, output_dir, conf=0.25):
    """
    YOLOv8 모델로 테스트 이미지를 예측하고 결과를 저장합니다.
    
    Args:
        model: 학습된 YOLOv8 모델
        test_images_dir (str): 테스트 이미지 디렉토리
        output_dir (str): 출력 디렉토리
        conf (float): 신뢰도 임계값
    """
    # 현재 날짜 및 시간으로 저장 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, "results", f"predict_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 테스트 이미지 가져오기
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not test_images:
        print("테스트할 이미지가 없습니다.")
        return None
    
    # 랜덤하게 10개 이미지 선택 (또는 모든 이미지가 10개 이하면 모두 사용)
    test_sample = random.sample(test_images, min(10, len(test_images)))
    
    # 예측 실행 - 명시적으로 프로젝트 디렉토리와 이름 지정
    for img_path in test_sample:
        # 이미지 이름 추출
        img_name = os.path.basename(img_path)
        
        # 예측 실행
        results = model.predict(
            source=img_path, 
            conf=conf,
            save=True,
            project=os.path.join(output_dir, "runs"),
            name=f"predict_{timestamp}",
            device='mps'  # MPS 장치 사용 설정
        )
        
        # 결과 이미지 저장
        try:
            # 예측 결과 디렉토리
            pred_result_dir = os.path.join(output_dir, "runs", f"predict_{timestamp}")
            
            # 결과 이미지 경로
            result_img_path = None
            for file in os.listdir(pred_result_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')) and img_name in file:
                    result_img_path = os.path.join(pred_result_dir, file)
                    break
            
            # 결과 이미지 복사
            if result_img_path and os.path.exists(result_img_path):
                save_img_path = os.path.join(results_dir, f"result_{img_name}")
                shutil.copy2(result_img_path, save_img_path)
                print(f"예측 결과 이미지 저장됨: {save_img_path}")
        except Exception as e:
            print(f"결과 이미지 저장 중 오류 발생: {e}")
        
        print(f"이미지 예측 완료: {img_path}")
    
    print(f"총 {len(test_sample)}개 이미지에 대한 예측이 완료되었습니다.")
    print(f"모든 결과는 {results_dir}에 저장되었습니다.")
    return results_dir


def main():
    """
    메인 함수 - YOLOv8 모델 학습 및 테스트 실행
    """
    # 경로 설정
    project_dir = "./"
    image_dir = os.path.join(project_dir, "sample_data/원천데이터")
    label_dir = os.path.join(project_dir, "sample_data/yolo_output")
    
    # 결과 저장을 위한, 사용자 지정 출력 디렉토리 (Desktop에 저장)
    user_home = os.path.expanduser("~")
    desktop_dir = os.path.join(user_home, "Desktop")
    output_base_dir = os.path.join(desktop_dir, "YOLO_RESULTS")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 현재 날짜 및 시간으로 저장 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"결과가 다음 위치에 저장됩니다: {output_dir}")
    
    # YOLOv8 데이터셋 준비
    dataset_dir = prepare_data_for_yolov8(
        project_dir=project_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=os.path.join(output_dir, "dataset")
    )
    
    # 데이터셋 YAML 파일 경로
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    
    # YOLOv8 모델 학습
    model = train_yolov8(
        dataset_yaml=dataset_yaml,
        output_dir=output_dir,
        model_size="n",  # nano 모델 (가장 작고 빠름)
        epochs=10,
        batch_size=16,
        img_size=640
    )
    
    # 모델 검증
    validate_yolov8(model, dataset_yaml, output_dir)
    
    # 테스트 이미지 예측 및 시각화
    test_images_dir = os.path.join(dataset_dir, "test", "images")
    results_dir = test_yolov8_with_visualization(model, test_images_dir, output_dir)
    
    # 결과 요약 출력
    print("\n" + "="*50)
    print("YOLOv8 모델 학습 및 테스트 완료!")
    print("="*50)
    print(f"모든 결과는 다음 위치에 저장되었습니다: {output_dir}")
    print(f"  - 데이터셋: {dataset_dir}")
    print(f"  - 학습된 모델: {os.path.join(output_dir, 'models')}")
    print(f"  - 테스트 결과: {results_dir}")


if __name__ == "__main__":
    import sys
    main()