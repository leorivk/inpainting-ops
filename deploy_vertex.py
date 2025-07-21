import os
from google.cloud import aiplatform

# 환경 변수에서 설정 값 가져오기
project_id = os.environ["GCP_PROJECT_ID"]
region = os.environ["GCP_REGION"]
repository = os.environ["ARTIFACT_REGISTRY_REPO"]
image_name = os.environ["IMAGE_NAME"]
image_tag = os.environ["IMAGE_TAG"]
model_display_name = "inpainting-model"
endpoint_display_name = "inpainting-endpoint"
machine_type = "n1-standard-4"
accelerator_type = "NVIDIA_TESLA_T4"
accelerator_count = 1

# Vertex AI 클라이언트 초기화
aiplatform.init(project=project_id, location=region)

# 배포할 Docker 이미지 URI
image_uri = f"{region}-docker.pkg.dev/{project_id}/{repository}/{image_name}:{image_tag}"
print(f"Deploying image: {image_uri}")

# 1. 기존에 같은 이름의 모델이 있는지 확인하고, 있다면 기존 버전 삭제
models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
if models:
    model_to_delete = models[0]
    print(f"Deleting existing model version: {model_to_delete.resource_name}")
    model_to_delete.delete()
    print("Old model deleted.")

# 2. Vertex AI에 모델 업로드
print("Uploading model to Vertex AI...")
model = aiplatform.Model.upload(
    display_name=model_display_name,
    serving_container_image_uri=image_uri,
    serving_container_predict_route="/inpaint",
    serving_container_health_route="/", # 헬스 체크용 루트 추가 필요
    serving_container_ports=[8080],
)
print(f"Model uploaded: {model.resource_name}")

# 3. 엔드포인트 생성 또는 가져오기
print("Creating or getting endpoint...")
endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')
if endpoints:
    endpoint = endpoints[0]
    print(f"Using existing endpoint: {endpoint.resource_name}")
else:
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    print(f"Endpoint created: {endpoint.resource_name}")

# 4. 모델을 엔드포인트에 배포
print("Deploying model to endpoint...")
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=model_display_name,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    traffic_split={"0": 100},
    min_replica_count=1, # 비용 절약을 위해 1로 설정, 트래픽 없을 시 0으로 자동 축소 가능
    max_replica_count=1,
)
print("Deployment to Vertex AI Endpoint completed successfully!")