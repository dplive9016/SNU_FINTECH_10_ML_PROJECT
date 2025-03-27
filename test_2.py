from ultralytics import YOLO
import cv2
import os

def predict_objects(model_path, image_path, output_dir='results', conf=0.25):
    """
    YOLOv8 모델을 사용하여 이미지에서 객체를 감지합니다.
    
    Args:
        model_path (str): 학습된 모델 파일 경로 (best.pt)
        image_path (str): 예측할 이미지 파일 경로 또는 디렉토리
        output_dir (str): 결과 이미지를 저장할 디렉토리
        conf (float): 신뢰도 임계값 (0.0 ~ 1.0)
    """
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 클래스 이름 설정
    class_names = [
        "졸음/피로 및 무기력",
        "운전자 방해 및 위험 행동",
        "물건 사용/조작 관련",
        "신체 동작 및 접촉/상호작용"
    ]
    
    # 이미지 경로가 디렉토리인 경우
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    else:
        image_files = [image_path]
    
    # 각 이미지에 대해 예측 수행
    for img_path in image_files:
        print(f"이미지 예측 중: {img_path}")
        
        # 예측 수행
        results = model.predict(
            source=img_path,
            conf=conf,
            save=True,
            project=output_dir,
            name='predictions',
            device='mps'  # MPS를 사용하려면 'mps', GPU는 0, CPU는 'cpu'
        )
        
        # 원본 이미지 로드
        img = cv2.imread(img_path)
        
        # 각 결과에 대해 처리
        for result in results:
            boxes = result.boxes  # BoundingBoxes 객체 가져오기
            
            # 각 박스마다 정보 출력 및 이미지에 표시
            for box in boxes:
                # 좌표 정보 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 클래스 ID 및 신뢰도
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # 클래스 이름 가져오기
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                
                print(f"감지된 객체: {class_name}, 신뢰도: {conf:.2f}, 좌표: [{x1}, {y1}, {x2}, {y2}]")
        
        print(f"결과 이미지가 {output_dir}/predictions 폴더에 저장되었습니다.")
    
    print("모든 이미지 예측 완료!")

# 실행 예시
if __name__ == "__main__":
    # 모델 경로, 이미지 경로, 출력 디렉토리 설정
    model_path = "/Users/admin/Desktop/YOLO_RESULTS/run_20250327_105843/models/train_20250327_105846/best.pt"  # 학습된 모델 파일 경로
    image_path = "/Users/admin/Desktop/Cursor/ML_DRIVER_PROJECT/SNU_FINTECH_10_ML_PROJECT/SGA2101509S2205IMG0005.jpg"   # 예측할 이미지 파일 또는 디렉토리 경로
    output_dir = "results"          # 결과 저장 디렉토리
    
    # 예측 실행
    predict_objects(model_path, image_path, output_dir, conf=0.25)