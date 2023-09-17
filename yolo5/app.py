import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
import pymongo

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ['AWS_REGION']

# Create the 'temp' directory if it doesn't exist
temp_dir = 'temp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

s3 = session.client('s3')

images_bucket = os.environ['BUCKET_NAME']
database_name = "mydb"
collection_name = "predictions"
mongodb_uri = f'mongodb://mongo1:27017,mongo2:27018,mongo3:27019/{database_name}?replicaSet=myReplicaSet'
client = pymongo.MongoClient(mongodb_uri)
db = client[database_name]
collection = db[collection_name]

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to
    # identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.
    original_img_path = f"temp/{img_name}"

    try:
        s3.download_file(images_bucket, img_name, original_img_path)
    except Exception as e:
        logger.error(f"Failed to download image from S3: {str(e)}")
        return f"Failed to download image from S3: {str(e)}", 500

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels The predicted image typically includes bounding boxes
    # drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    # Upload the predicted image to S3
    try:
        s3.upload_file(str(predicted_img_path), images_bucket, f"predicted/{prediction_id}/{img_name}")
    except Exception as e:
        logger.error(f"Failed to upload predicted image to S3: {str(e)}")

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')

    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
        logger.info(f'prediction: {prediction_id}/{original_img_path}. created prediction summery')
        insert_id = collection.insert_one(prediction_summary)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. written to mongodb cluster. ID:{insert_id}')
        prediction_summary.pop('_id')
        logger.info(f'prediction: {prediction_id}/{original_img_path}. current pred_sum: {prediction_summary}')

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
