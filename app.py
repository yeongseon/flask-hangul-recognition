import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

MODEL_NAME = "vit_b_32"
MODEL_FILE = f"models/{MODEL_NAME}.pth"
LABEL_FILE = "labels/256-common-hangul.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(
    model_name, num_classes, feature_extract, use_pretrained=True, checkpoint_path=None
):
    # 모델을 선택하고 초기화합니다.

    if model_name == "vgg19":
        """VGG19"""
        model = models.vgg19(pretrained=use_pretrained)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vit_b_32":
        """Vision Transformer B-32"""
        model = models.vit_b_32(pretrained=use_pretrained)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        # ViT의 분류기 부분을 교체합니다.
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    # 체크포인트에서 모델 가중치를 로드할 경우
    if checkpoint_path:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device("cpu"))
        )

    return model


def predict_image(model, image_path, transform, device):
    # 이미지 불러오기
    image = Image.open(image_path).convert("L")

    # 이미지 전처리
    image = transform(image).unsqueeze(0)  # 차원 추가 (배치 차원)
    image = image.to(device)

    # 모델 추론 모드 설정 및 예측
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


model = initialize_model(
    MODEL_NAME,
    num_classes=256,
    feature_extract=True,
    use_pretrained=False,
    checkpoint_path=MODEL_FILE,
)
model = model.to(DEVICE)


def get_label_at_index(label_file_path, index):
    try:
        with open(label_file_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i == index:
                    return line.strip()  # 공백 및 줄바꿈 제거
    except FileNotFoundError:
        print(f"The file {label_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None  # 파일을 찾지 못하거나 다른 에러 발생 시 None 반환


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    # Get the filename of the most recently uploaded file, if any
    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    uploaded_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(app.config["UPLOAD_FOLDER"], x)),
        reverse=True,
    )
    latest_file = uploaded_files[0] if uploaded_files else None

    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # 이미지 분류
        classification_result = predict_image(model, file_path, transform, DEVICE)
        print(classification_result)
        print(get_label_at_index(LABEL_FILE, classification_result))
        classification_result = get_label_at_index(LABEL_FILE, classification_result)

        return render_template(
            "index.html",
            latest_file=file.filename,
            classification_result=classification_result,
        )

        # 결과를 index 뷰로 전달
        # return redirect(url_for('index', classification_result=classification_result))

    return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
